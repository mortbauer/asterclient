import os
import sys
import zipfile
import shutil
import yaml
import time
import glob
import imp
import tempfile
import argparse
import pkgutil
import subprocess
import signal
import multiprocessing
import threading
import collections
import logging

RUNPY_TEMPLATE = """#!{aster}
#coding=utf-8
import sys
sys.path.insert(0,'{bibpyt}')
sys.path.extend({syspath})
from Execution.E_SUPERV import SUPERV
from Execution.E_Core import getargs
supervisor = SUPERV()
res = supervisor.main(coreopts=getargs({arguments}),params={params})
sys.exit(res)"""

RUNSH_TEMPLATE = """#!/usr/bin/sh
export LD_LIBRARY_PATH="{LD_LIBRARY_PATH}:${{LD_LIBRARY_PATH}}"
exec {runpy}
"""

logger = logging.getLogger('asterclient')

def get_code_aster_error(filename):
    res = []
    record = False
    with open(filename,'r') as f:
        for line in f:
            if line.startswith('>>') and not record:
                record = True
            elif line.startswith('>>') and record:
                record = False
            elif record:
                res.append(line)
    return res

def make_pasrer():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-level',default='INFO',
            help='specify the logging level')
    subparsers = parser.add_subparsers(dest='action')
    infoparser = subparsers.add_parser('info')
    infoparser.add_argument('-p','--profile',required=True,
            help='specify profile file',type=argparse.FileType('r'))
    infoparser.add_argument('--studies',action='store_true',
            help='list the available studies of the profile')
    infoparser.add_argument('--calculations',action='store_true',
            help='list the available calculations of the profile')
    runparser = subparsers.add_parser('run')
    runparser.add_argument('-p','--profile',required=True,
            help='specify profile file',type=argparse.FileType('r'))
    runparser.add_argument('-s','--study',nargs='*',
            help='run the these studies')
    runparser.add_argument('-c','--calculation',nargs='*',
            help='run the these calculations')
    runparser.add_argument('--clean',action='store_true',
            help='cleans existing workdir and resultdir')
    runparser.add_argument('--workdir',
            help='work directory for calculation')
    runparser.add_argument('--parallel',action='store_true',
            help='don\'t dispatch but run parallel')
    runparser.add_argument('--max-parallel',required=False,default=10,
            help='limit the number of parallel processes',type=int)
    runparser.add_argument('--hide-aster',action='store_true',
            help='hide the output of code aster')

    runparser.add_argument('--bibpyt',
        help="path to Code_Aster python source files")
    runparser.add_argument('--memjeveux',
        help="maximum size of the memory taken by the execution (in Mw)")
    runparser.add_argument('--memory',
        help="maximum size of the memory taken by the execution (in MB)")
    runparser.add_argument('--tpmax',
        help="limit of the time of the execution (in seconds)")
    runparser.add_argument('--max_base',
        help="limit of the size of the results database")
    runparser.add_argument('--dbgjeveux',action='store_true',
        help="maximum size of the memory taken by the execution in Mw")
    runparser.add_argument('--mode',
        help="execution mode (interactive or batch)")
    runparser.add_argument('--interact', action='store_true', default=False,
        help="as 'python -i' works, it allows to enter commands after the "
                "execution of the command file.")
    runparser.add_argument('--rep_outils',
        help="directory of Code_Aster tools (ex. $ASTER_ROOT/outils)")
    runparser.add_argument('--rep_mat',
        help="directory of materials properties")
    runparser.add_argument('--rep_dex',
        help="directory of external datas (geometrical datas or properties...)")
    runparser.add_argument('--suivi_batch',action='store_true',default=True,
        help="force to flush of the output after each line")
    runparser.add_argument('--verif', action='store_true', default=False,
        help="only check the syntax of the command file is done")
    runparser.add_argument('--prepare', action='store_true', default=False,
        help="only prepare everything")
    return parser

def execute_one(calculations):
    for calc in calculations:
        try:
            calc.run()
        except KeyboardInterrupt:
            calc.subprocess.kill()
            break


class Consumer(multiprocessing.Process):

    def __init__(self, task_queue, kill_event,counter,counter_lock):
        # let us ignore interupts
        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue
        self.kill_event = kill_event
        self.counter = counter
        self.counter_lock = counter_lock

    def run(self):
        proc_name = self.name
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        while True and not self.kill_event.is_set():
            try:
                next_task = self.task_queue.get(timeout=4)
            except:
                # queue is empty, stop
                with self.counter_lock:
                    if self.counter.value == 0:
                        break
                    else:
                        continue
            logger.info('%s starts: %s' % (proc_name, next_task))
            answer = next_task(self.kill_event)
            with self.counter_lock:
                self.counter.value -= 1
            for calc in next_task.run_after:
                self.task_queue.put(calc)
            logger.info('%s finnished: %s' % (proc_name, next_task))

class AsterClientException(Exception):
    pass

class Config(object):
    def __init__(self,asterclient,options):
        for key in asterclient.profile:
            if not hasattr(self,key):
                setattr(self,key,asterclient.profile[key])
        doptions = vars(options)
        for key in doptions:
            if not hasattr(self,key):
                setattr(self,key,doptions[key])

    def __getitem__(self,key):
        return getattr(self,key)

class AsterClient(object):
    def __init__(self,options):
        self.options = options
        self._remove_at_exit = []
        self.profile = self._load_profile()
        self.basepath = os.path.abspath(self.profile.get('srcdir'))
        self.studies = self._load_studies()

    def init(self):
        self._sanitize_profile(self.profile)
        self._prepare_paths_for_run()
        self._check_settings()
        self.config = Config(self,self.options)

    def _abspath(self,path):
        if os.path.isabs(path):
            return path
        else:
            return os.path.abspath(os.path.join(self.basepath,path))

    def _load_profile(self):
        # get default profile
        if not os.environ.get('ASTER_ROOT'):
            raise AsterClientException(
                'the ASTER_ROOT environment variable must be set')
        profile = yaml.load(pkgutil.get_data(
            __name__,os.path.join('data','defaults.yml')))
        # read provided profile
        try:
            profile.update(yaml.load(self.options.profile.read()))
        except Exception as e:
            raise AsterClientException(
                'the profile {0} couldn\'t be parsed:\n\n{1}'
                .format(self.options.profile.name,e))
        return profile

    def _sanitize_profile(self,profile):
        # make paths absolute
        for pathkey in ['bibpyt','cata','elements','rep_mat','rep_dex','aster']:
            profile[pathkey] = os.path.join(
                profile['aster_root'],profile['version'],profile[pathkey])
        if not os.path.isabs(profile['rep_outils']):
            profile['rep_outils'] = os.path.join(
                profile['aster_root'],profile['rep_outils'])
        # populate the commandline options into the profile
        for key in ['bibpyt','memjeveux','memory','tpmax','max_base',
                'dbgjeveux','mode','interact','rep_outils','rep_mat',
                'rep_dex','suivi_batch','verif','workdir']:
            if getattr(self.options,key) != None:
                profile[key] = getattr(self.options,key)
        # create a named bound for the calculations
        self.calculations = {calc['name']:calc for calc in self.profile['calculations']}

    def _load_studies(self):
        if 'distributionfile' in self.profile:
            distributionfile = self._abspath(self.profile['distributionfile'])
            distributionfilename = os.path.splitext(os.path.basename(distributionfile))[0]
            try:
                studies = imp.load_source(distributionfilename,distributionfile).parameters
            except Exception as e:
                    raise AsterClientException('couldn\'t import distributionfile, %s'%e)
            self.distributionfile = distributionfile
        else:
            studies = [{'name':'main'}]
            self.distributionfile = None
        self.profile['distributionfile'] = self.distributionfile
        return studies

    def _check_settings(self):
        # check if all minimum needed keys are available
        if not 'calculations' in self.profile:
            raise AsterClientException('you need to specify at least one calculation')
        self._calculation_names = []
        self._study_names = []
        for calculation in self.profile['calculations']:
            self._sanitize_calculation_options(calculation)
        # check the studie
        study_keys = set()
        i = 0
        for study in self.studies:
            self._sanitize_study_options(study)
            study_keys.add(tuple(study.keys()))
            study['studynumber'] = i
            i += 1

        # check if all studies have the same keys
        if len(study_keys) > 1:
            raise AsterClientException('all studies need to have the same keys')

    def _sanitize_calculation_options(self,calculation):
        # test calculations specified in the profile
        name = calculation.get('name','')
        if not name:
            raise AsterClientException('no calcualtion name specified')
        if name in self._calculation_names:
            raise AsterClientException('calculation names {0} isn\'t unique'
                                       .format(name))
        else:
            self._calculation_names.append(name)

        commandfile = calculation.get('commandfile','')
        if not commandfile:
            raise AsterClientException('no commandfile specified for "{0}"'
                                       .format(name))
        calculation['commandfile'] = self._abspath(commandfile)
        if 'inputfiles' in calculation:
            for i,fpath in enumerate(calculation['inputfiles']):
                calculation['inputfiles'][i] = self._abspath(fpath)

    def _sanitize_study_options(self,study):
        # test the studies specified in the distributionfile
        name = study.get('name','')
        if not name:
            raise AsterClientException('every study needs a name')
        if name in self._study_names:
            raise AsterClientException('study names {0} isn\'t unique'.format(name))
        else:
            self._study_names.append(name)

        meshfile = study.get('meshfile',self.profile.get('meshfile'))
        if not meshfile:
            raise AsterClientException('no meshfile specified for "{0}"'.format(name))
        study['meshfile'] = self._abspath(meshfile)

    def get_calculations_to_run(self):
        # get the calculations which should be run
        if self.options.calculation:
            calculations = []
            calcnames = [calc['name'] for calc in self.profile['calculations']]
            for x in self.options.calculation:
                try:
                    # get by index
                    key = int(x)
                except:
                    key = None
                if key == None:
                    try:
                        key = calcnames.index(x)
                    except:
                        raise AsterClientException('there is no calculation "{0}"'.format(x))
                calculations.append(self.profile['calculations'][key])
        else:
            # take all if none is specified
            calculations = self.profile['calculations']
        return calculations

    def get_studies_to_run(self):
        # get the studies which should be run
        if self.options.study:
            studies = []
            studynames = [study['name'] for study in self.studies]
            for x in self.options.study:
                try:
                    # get by index
                    key = int(x)
                except:
                    key = None
                if key == None:
                    try:
                        key = studynames.index(x)
                    except:
                        raise AsterClientException('ther is no study "{0}"'.format(x))
                studies.append(self.studies[key])
        else:
            studies = self.studies
        return studies

    def _prepare_paths_for_run(self):
        """ set up the paths
        """
        self.outdir = self._abspath(self.profile.get('outdir','results'))
        self.profile['outdir'] = self.outdir
        # setup the workdir
        workdir = self.profile.get('workdir',self.options.workdir)
        if not workdir:
            workdir = tempfile.mkdtemp(
                '_asterclient',self.profile.get('project','tmp'))
            self._remove_at_exit.append(workdir)
        else:
            workdir = self._abspath(workdir)
            try:
                os.makedirs(workdir)
            except:
                pass
        self.workdir = workdir
        self.profile['workdir'] = self.workdir

    def _create_executions(self):
        executions = []
        count = 0
        for study in self.studies_to_run:
            tmp = {}
            for calc in self.calculations_to_run:
                need = calc.get('poursuite')
                tmp[calc['name']] = Calculation(
                    '%s:%s'%(study['name'],calc['name']),
                    self.config,study,calc,need)
                count += 1
            for calc in tmp.values():
                if calc.needs and calc.needs in tmp:
                    tmp[calc.needs].run_after = calc
                else:
                    executions.append(calc)
        return executions,count

    def run(self):
        # do whatever has to be done
        if self.options.action == 'info':
            self.info_studies()
            self.info_calculations()
        elif self.options.action == 'run':
            self.init()
            self._run()

    def _run_parallel(self):
        kill_event = multiprocessing.Event()
        def shutdown(sig,frame):
            logger.warn('will shutdown all processes, interrupt')
            kill_event.set()
        signal.signal(signal.SIGINT,shutdown)
        #task_queue = collections.deque()
        task_queue = multiprocessing.Queue()
        counter_lock = multiprocessing.Lock()
        counter = multiprocessing.Value('i',(self.num_executions))
        # Enqueue jobs
        for calc in self.executions:
            task_queue.put(calc)
        # Start consumers
        consumers = []
        for i in range(self.options.max_parallel):
            c = Consumer(task_queue,kill_event,counter,counter_lock)
            consumers.append(c)
            c.start()
        # Wait for all of the tasks to finish
        task_queue.close()

    def _run_sequential(self,calcs):
        for calc in calcs:
            logger.info('start %s' % calc)
            calc()
            self._run_sequential(calc.run_after)

    def _run(self):
        self.calculations_to_run = self.get_calculations_to_run()
        self.studies_to_run = self.get_studies_to_run()
        self.executions,self.num_executions = self._create_executions()
        if self.options.parallel and not self.options.prepare:
            self._run_parallel()
        elif self.options.parallel and self.options.prepare:
            raise AsterClientException('parallel preparation not supported')
        else:
            self._run_sequential(self.executions)

    def info_studies(self):
        """ print the available studies
        of the given profile file"""
        print('\navailable parametric studies:\n')
        i = 0
        for key in self.studies:
            print('\t{0}: {1}\n'.format(i,key['name']))
            i +=1

    def info_calculations(self):
        """ print the available calculations
        of the given profile file"""
        print('\navailable calculations:\n')
        for i,calc in enumerate(self.profile['calculations']):
            print('\t{0}: {1}\n'.format(i,calc['name']))

class Calculation(object):
    # unfortunately POURSUITE only works in a new process,
    # therefore let us call everything in a knew process
    def __init__(self,name,config,study,calculation,needs):
        self.config = config
        self.study = study
        self.calculation = calculation
        # my needs calculation, needed if we resume
        self.needs = needs
        self.name = name
        self._initiated = False
        self._processing = False
        self.success = False
        self.finnished = False
        self._run_after = []

    def __str__(self):
        return '<Calculation: %s>'%self.name

    @property
    def run_after(self):
        return self._run_after

    @run_after.setter
    def run_after(self,calc):
        self._run_after.append(calc)

    def _prepare_paths(self):
        self.buildpath = os.path.join(
            self.config.workdir,self.study['name'],self.calculation['name'])
        self.outputpath = os.path.join(
            self.config.outdir,self.study['name'],self.calculation['name'])
        # make sure buildpath exists and is clean
        try:
            os.makedirs(self.buildpath)
        except:
            if self.config.clean:
                logger.debug('cleaning buildpath "{0}"'.format(self.buildpath))
                shutil.rmtree(self.buildpath)
                os.makedirs(self.buildpath)
            else:
                logger.warn('buildpath "{0}" exists and holds data'.format(self.buildpath))
        # make sure output directory exists and is clean
        try:
            os.makedirs(self.outputpath)
        except:
            if self.config.clean:
                logger.debug('cleaning outputpath "{0}"'.format(self.outputpath))
                shutil.rmtree(self.outputpath)
                os.makedirs(self.outputpath)
            else:
                logger.warn('outputpath "{0}" exists and holds data'.format(self.outputpath))
        self.infofile = os.path.join(self.buildpath,'fort.6')

    def _copy_files(self):
        # copy the elements catalog
        shutil.copyfile(
            self.config.elements,os.path.join(self.buildpath,'elem.1'))
        # copy meshfile
        shutil.copyfile(self.study['meshfile'],
                        os.path.join(self.buildpath,'fort.20'))
        # copy commandfile
        shutil.copyfile(
            self.calculation['commandfile'],
            os.path.join(self.buildpath,'fort.1'))
        # if calculation is a continued one copy the results from the
        if 'poursuite' in self.calculation and self.needs:
            glob1 = os.path.join(self.config['outdir'],self.study['name'],
                                 self.needs,'glob.1.zip')
            pick1 = os.path.join(self.config['outdir'],self.study['name'],
                                 self.needs,'pick.1.zip')
            try:
                logger.debug('try to copy glob.1.zip from %s'%glob1)
                with zipfile.ZipFile(glob1,'r',allowZip64=True) as zipf:
                    zipf.extractall(path=self.buildpath)
                logger.debug('try to copy pick.1.zip from %s'%pick1)
                with zipfile.ZipFile(pick1,'r',allowZip64=True) as zipf:
                    zipf.extractall(path=self.buildpath)
            except Exception as e:
                raise AsterClientException(
                    'failed to copy glob.1 and pick.1 from "{0}"'.format(self.needs))


    def _createresultfiles(self):
        # the resultfiles need already to be created for fortran
        # create a list of files which need to be copied to the
        # resultdirectory
        resultfiles = {}
        if 'resultfiles' in self.calculation:
            for f in self.calculation['resultfiles']:
                for key in f:
                    # result files for acces through fortran
                    if type(f[key]) == int:
                        name = 'fort.%s' % f[key]
                        with open(os.path.join(self.buildpath,name),'w') as f:
                            resultfiles[key] = name
                    else:
                        resultfiles[key] = f[key]
        self.resultfiles = resultfiles

    def _copy_additional_inputfiles(self):
        if 'inputfiles' in self.calculation:
            for f in self.calculation['inputfiles']:
                try:
                    shutil.copyfile(f,os.path.join(self.buildpath,os.path.basename(f)))
                except:
                    logger.error('failed to copy input file "{0}"'.format(f))

    def _create_runpy(self):
        arguments = ['supervisor',
                '--bibpyt', self.config['bibpyt'],
                '--commandes','fort.1',
                '--mode',self.config['mode'],
                '--rep_outils',self.config['rep_outils'],
                '--rep_mat',self.config['rep_mat'],
                '--rep_dex',self.config['rep_dex'],
                '--bibpyt',self.config['bibpyt'],
                '--tpmax',str(self.config['tpmax']),
                '--memjeveux',str(self.config['memjeveux'])]
        if self.config['memory']:
            arguments.extend(['--memory',self.config['memory']])
        if self.config['max_base']:
            arguments.extend(['--max_base',self.config['max_base']])
        if self.config['suivi_batch']:
            arguments.append('--suivi_batch')

        runpy = RUNPY_TEMPLATE.format(aster=self.config.aster,
                   bibpyt=self.config.bibpyt,
                   syspath=sys.path,
                   arguments=arguments,
                   params=self.study)

        self.runpy_path = os.path.join(self.buildpath,'run.py')
        with open(self.runpy_path,'w') as f:
            f.write(runpy)
        os.chmod(self.runpy_path,0777)

    def _create_runsh(self):
        runsh = RUNSH_TEMPLATE.format(LD_LIBRARY_PATH=self.buildpath,runpy=self.runpy_path)

        self.runsh_path = os.path.join(self.buildpath,'run.sh')
        with open(self.runsh_path,'w') as f:
            f.write(runsh)
        os.chmod(self.runsh_path,0777)

    def _run_bashed(self):
        #signal.signal(signal.SIGINT,self.shutdown)
        if self.config.parallel and not self.config.hide_aster:
            tee = '| tee {0}; exit $PIPESTATUS'.format(self.infofile)
        else:
            tee = '2&> {0};exit $PIPESTATUS'.format(self.infofile)
        bashscript = '{runsh} {tee}'.format(runsh=self.runsh_path,tee=tee)
        self.subprocess = subprocess.Popen(['bash','-c',bashscript],cwd=self.buildpath)
        wait = True
        while wait:
            if self.subprocess.poll() != None:
                wait = False
            elif self._kill_event and self._kill_event.is_set():
                self.subprocess.kill()
                wait = False
            else:
                time.sleep(2)

    def init(self):
        if not self._initiated:
            self._prepare_paths()
            self._copy_files()
            self._copy_additional_inputfiles()
            self._createresultfiles()
            self._create_runpy()
            self._create_runsh()
            self._initiated = True

    def __call__(self,kill_event=None):
        self._kill_event = kill_event
        if not self._processing:
            try:
                self._processing = True
                if self.config.prepare:
                    self.init()
                    logger.info('cd to "{0}" and run "run.sh" '.format(self.buildpath))
                else:
                    self.init()
                    self._run_bashed()
                    self._copyresults()
                    if self.subprocess.returncode == 0:
                        self.success = True
            except:
                logger.exception('internal error')
            finally:
                self._processing = False
                self.finnished = True

    def _copyresults(self):
        # try to copy results even if errors occured
        for name,fpath in self.resultfiles.items():
            for f in glob.glob(os.path.join(self.buildpath,fpath)):
                outname = os.path.basename(f)
                if os.path.getsize(f) == 0 and self.subprocess.returncode == 0:
                    logger.warn('result file "{0}" is empty'.format(outname))
            self._copyresult(f,os.path.join(self.outputpath,outname))

        # copy additional inputfiles as well
        for f in self.calculation.get('inputfiles',[]):
            self._copyresult(f,
                os.path.join(self.outputpath,os.path.basename(f)))

        # copy the standard result files
        self._copyresult(
            os.path.join(self.buildpath,'fort.6'),
            os.path.join(self.outputpath,self.calculation['name']+'.mess')
        )
        # copy the commandfile as well as the parameters and the mesh
        self._copyresult(
            os.path.join(self.buildpath,'fort.1'),
            os.path.join(self.outputpath,os.path.basename(
                self.calculation['commandfile'])),
        )
        self._copyresult(
            os.path.join(self.buildpath,'fort.20'),
            os.path.join(self.outputpath,os.path.basename(
            self.config.meshfile)),
        )
        if self.config.distributionfile:
            self._copyresult(
                self.config.distributionfile,
                os.path.join(self.outputpath,os.path.basename(
                    self.config.distributionfile))
            )
        # copy the zipped base
        self._copyresult(
            os.path.join(self.buildpath,'glob.1'),
            os.path.join(self.outputpath,'glob.1.zip'),zipped=True
        )
        self._copyresult(
            os.path.join(self.buildpath,'pick.1'),
            os.path.join(self.outputpath,'pick.1.zip'),zipped=True
        )

    def _copyresult(self,fromfile,tofile,zipped=False):
        try:
            if not zipped:
                shutil.copyfile(fromfile,tofile)
            else:
                zipf = zipfile.ZipFile(tofile,'w',allowZip64=True)
                zipf.write(fromfile,arcname=os.path.basename(fromfile),
                           compress_type=zipfile.ZIP_DEFLATED)
                zipf.close()
        except Exception as e:
            if self.subprocess.returncode == 0:
                raise e
            else:
                logger.info('ignore exception for copying result "{0}"'
                            .format(os.path.basename(fromfile)))

def main(argv=None):
    if not argv:
        argv = sys.argv[1:]
    parser = make_pasrer()
    options = parser.parse_args(argv)
    formatter = logging.Formatter('%(name)s: %(levelname)s: %(message)s')
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    try:
        logger.setLevel(options.log_level.upper())
    except:
        logger.setLevel('INFO')
    if options.log_level.upper() == 'DEBUG' and not options.parallel:
        import debug
    asterclient = AsterClient(options)
    try:
        asterclient.run()
    except KeyboardInterrupt:
        logger.error('killed all calculations through interrupt')

if '__main__' == __name__:
    main()
