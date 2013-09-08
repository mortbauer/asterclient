import os
import sys
from zipfile import ZipFile
import shutil
import yaml
import time
import glob
import imp
import atexit
import tempfile
import argparse
import pkgutil
import subprocess
import threading
import termcolor
import signal
import multiprocessing
import logging
import random
import string
#import ipdb
import debug

logger = logging.getLogger('asterclient')
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel('INFO')

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

class AsterClientException(Exception):
    pass

def make_pasrer():
    parser = argparse.ArgumentParser()
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
    runparser.add_argument('--force',action='store_true',
            help='overwrites existing files and directories')
    runparser.add_argument('--workdir',
            help='work directory for calculation')
    runparser.add_argument('--sequential',action='store_true',
            help='don\'t dispatch but run sequential')

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

def info_studies(parameters,studyname=None):
    """ print the available studies
    of the given profile file"""
    if not studyname:
        print('\navailable parametric studies:\n')
        i = 0
        for key in parameters:
            print('\t{0}: {1}\n'.format(i,key['name']))
            i +=1
    else:
        print('\nparameters of study {0}:\n'.format(studyname))
        print(parameters[x][1] for x in parameters if parameters[x] == studyname)

def info_calculations(profilename,profile):
    """ print the available calculations
    of the given profile file"""
    print('\navailable calculations:\n')
    for i in range(len(profile['calculations'])):
        print('\t{0}: {1}\n'.format(i,profile['calculations'][i]['name']))


class Config(object):
    def __init__(self,asterclient,options):
        self.distributionfile = asterclient.distributionfile
        self.force = options.force
        for key in asterclient.profile:
            setattr(self,key,asterclient[key])

class AsterClient(object):
    def __init__(self,options):
        self.options = options
        self._remove_at_exit = []
        self.profile = self._load_profile()
        self._prepare_paths_for_run()
        self.studies = self._load_studies()
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
        self._sanitize_profile(profile)
        return profile

    def _sanitize_profile(self,profile):
        # make paths absolute
        for pathkey in ['bibpyt','cata','elements','rep_mat','rep_dex','aster']:
            if not os.path.isabs(profile[pathkey]):
                profile[pathkey] = os.path.join(
                    profile['aster_root'],profile['version'],profile[pathkey])
        if not os.path.isabs(profile['rep_outils']):
            profile['rep_outils'] = os.path.join(
                profile['aster_root'],profile['rep_outils'])
        self._merge_options_and_profile()

    def _load_studies(self):
        if 'distributionfile' in self.profile:
            distributionfile = self._abspath(self.profile['distributionfile'])
            distributionfilename = os.path.splitext(os.path.split(distributionfile)[-1])[0]
            try:
                studies = imp.load_source(distributionfilename,distributionfile).parameters
            except:
                    raise AsterClientException('couldn\'t import distributionfile')
            self.distributionfile = distributionfile
        else:
            studies = [{'name':'main'}]
            self.distributionfile = None
        return studies

    def _merge_options_and_profile(self):
        # populate the commandline options into the profile
        for key in ['bibpyt','memjeveux','memory','tpmax','max_base',
                'dbgjeveux','mode','interact','rep_outils','rep_mat',
                'rep_dex','suivi_batch','verif','workdir']:
            if getattr(self.options,key) != None:
                self.profile[key] = getattr(self.options,key)

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
            study_keys.add(set(self.study.keys()))
            study['studynumber'] = i
            i += 1

        # check if all studies have the same keys
        if len(study_keys) > 1:
            raise AsterClientException('all studies need to have the same keys')

    def _sanitize_calculation_options(self,calcualtion):
        # test calculations specified in the profile
        name = calculation.get('name','')
        if not name:
            raise AsterClientException('no calcualtion name specified')
        if name in self._calculation_names:
            raise AsterClientException('calculation names {0} isn\'t unique'.format(name))
        else:
            self._calculation_names.append(name)

        commandfile = calculation.get('commandfile','')
        if not commandfile:
            raise AsterClientException('no commandfile specified for "{0}"'.format(name))
        calculation['commandfile'] = self._abspath(commandfile)

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

    def _get_calculations(self):
        # get the calculations which should be run
        if self.options.calculation:
            calculations = []
            calcnames = [calculation['name'] for i in self.profile['calculations']]
            for x in self.options.calculation:
                try:
                    # get by index
                    key = int(x)
                except:
                    key = None
                if not key:
                    try:
                        key = calcnames.index(x)
                    except:
                        raise AsterClientException('ther is no calculation "{0}"'.format(x))
                calculations.append(self.profile['calculations'][key])
        else:
            # take all if none is specified
            calculations = self.profile['calculations']
        return calculations

    def _get_studies(self):
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
                if not key:
                    try:
                        key = studynames.index(x)
                    except:
                        raise AsterClientException('ther is no study "{0}"'.format(x))
                studies.append(self.studies[key])
        else:
            studies = self.studies
        return self.studies

    def _prepare_paths_for_run(self):
        """ set up the paths
        """
        self.basepath = self.profile.get('srcdir')
        self.outdir = self._abspath(self.profile.get('outdir','results'))
        # setup the builddir
        builddir = self.profile.get('workdir','')
        if not builddir:
            builddir = tempfile.mkdtemp(
                '_asterclient',self.profile.get('project','tmp'))
            self._remove_at_exit.append(builddir)
        else:
            builddir = self._abspath(builddir)
            try:
                os.makedirs(builddir)
            except:
                pass
        self.builddir = builddir

    def _setup_aster(self):
        """ a bit obsolete here """
        sys.path.append(profile['bibpyt'])
        # symlink cata to Cata/cata
        try:
            os.symlink(os.path.join(profile['cata'],'cata.py'),os.path.join(profile['bibpyt'],'Cata','cata.py'))
        except OSError as e:
            if e.errno == os.errno.EEXIST:
                logger.warn('couldn\'t symlink cata.py')
                #os.remove(os.path.join(profile['bibpyt'],'Cata','cata.py'))
                #os.symlink(os.path.join(profile['cata'],'cata.py'),os.path.join(profile['bibpyt'],'Cata','cata.py'))
        from Execution.E_SUPERV import SUPERV
        from Execution.E_Core import getargs
        self.supervisor = SUPERV()

    def _run_sequential(self):
        for study in self.studies_to_run:
            previous = None
            for calculation in self.calculations_to_run:
                calc = Calculation(self.config,study,calculation)
                calc.init()
                previous = calc.run()

    def _run_threaded(self):
        # start the process stepped
        counter = 0
        ncpus = 1 # multiprocessing.cpu_count()
        #ipdb.set_trace()
        for i in range(len(processes)):
            processes[i][1].start()
            counter += 1
            if counter == ncpus or counter == len(processes):
                time.sleep(0.1)
                for j in range(min(len(processes),ncpus)):
                    #print('trying to join process',j)
                    processes[j][1].join()
                counter = 0


    def run(self):
        # do whatever has to be done
        if self.options.action == 'info':
            info_studies(self.studies)
            info_calculations(self.options.profile.name,self.profile)
        elif self.options.action == 'run':
            self.run()

    def _run(self):
        self.calculations_to_run = self._get_calculations()
        self.studies_to_run = self._get_studies()
        if self.options.sequential:
            self._run_sequential()


class Calculation(object):
    # unfortunately POURSUITE only works in a new process,
    # therefore let us call everything in a knew process
    def __init__(self,config,study,calculation,previous=None):
        self.config = config
        self.study = study
        self.calculation = calculation
        # my previous calculation, needed if we resume
        self.previous = previous
        self.response = {}

    def _prepare_paths(self):
        self.buildpath = os.path.join(
            self.builddir,self.study['name'],self.calculation['name'])
        self.outputpath = os.path.join(
            self.outdir,self.study['name'],self.calculation['name'])
        # make sure buildpath exists and is clean
        try:
            os.makedirs(self.buildpath)
        except:
            if self.config.force:
                logger.info('cleaning buildpath "{0}"'.format(self.buildpath))
                shutil.rmtree(buildpath)
                os.makedirs(buildpath)
            else:
                logger.warn('buildpath "{0}" exists and holds data'.format(self.buildpath))
        # make sure output directory exists and is clean
        try:
            os.makedirs(self.outputpath)
        except:
            if self.config.force:
                logger.info('cleaning outputpath "{0}"'.format(self.outputpath))
                shutil.rmtree(outputpath)
                os.makedirs(outputpath)
            else:
                logger.warn('outputpath "{0}" exists and holds data'.format(self.outputpath))
        self.infofile = os.path.join(self.buildpath,'fort.6')

    def _copy_files(self):
        # copy the elements catalog
        shutil.copyfile(
            self.config.elements,os.path.join(self.buildpath,'elem.1'))
        # copy meshfile
        shutil.copyfile(self.study['meshfile'],
                        os.path.join(self.uildpath,'fort.20'))
        # copy commandfile
        shutil.copyfile(
            self.calculation['commandfile'],
            os.path.join(self.buildpath,'fort.1'))
        # if calculation is a continued one copy the results from the
        if 'poursuite' in self.calculation:
            with ZipFile(self.previous['glob.1.zip'],'r',allowZip64=True) as zipf:
                zipf.extractall(path=buildpath)
            with zipfile.ZipFile(previous['pick.1.zip'],'r',allowZip64=True) as zipf:
                zipf.extractall(path=buildpath)

    def _create_resultfiles(self):
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
                        with open(os.path.join(buildpath,name),'w') as f:
                            resultfiles[key] = name
                    else:
                        resultfiles[key] = f[key]
        self.resultfiles = resultfiles

    def _copy_additional_inputfiles(self):
        if 'inputfiles' in self.calculation:
            for f in self.calculation['inputfiles']:
                try:
                    fpath = os.path.join(self.config.srcdir,f)
                    shutil.copyfile(fpath,os.path.join(buildpath,os.path.split(f)[-1]))
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
        if profile['suivi_batch']:
            arguments.append('--suivi_batch')

        runpy = """#!{aster}
        #coding=utf-8
        import sys
        sys.path.insert(0,{bibpyt})
        sys.path.extend({syspath})
        from Execution.E_SUPERV import SUPERV
        from Execution.E_Core import getargs
        supervisor = SUPERV()
        res = supervisor.main(coreopts=getargs({arguments}),params={params})
        sys.exit(res)
        """.format(aster=self.config.aster,
                   bibpyt=self.config.bibpyt,
                   syspath=sys.path,
                   arguments=arguments,
                   params=self.study)

        self.runpy_path = os.path.join(self.buildpath,'run.py')
        with open(self.runpy_path,'w') as f:
            f.write(runpy)
        os.chmod(self.runpy_path,0777)

    def _create_runsh(self):
        runsh = """#!/usr/bin/sh
        export LD_LIBRARY_PATH="{LD_LIBRARY_PATH}"
        exec run.py
        """.format(LD_LIBRARY_PATH=os.pathsep + buildpath + os.environ['LD_LIBRARY_PATH'])

        self.runsh_path = os.path.join(self.buildpath,'run.py')
        with open(self.runsh_path,'w') as f:
            f.write(runsh)
        os.chmod(self.runsh_path,0777)

    def _run_bashed(self):
        if self.config.forground:
            tee = '| tee {0}; exit $PIPESTATUS'.format(self.infofile)
        else:
            tee = ''
        bashscript = '{runsh} {tee}'.format(runsh=self.runsh_path,tee=tee)
        self.result = subprocess.call(['bash','-c',bashscript])

    def _run_info(self):
        logger.info('code aster run "{study}:{calculation}" ended: {status}'.
                format(status = 'OK'if not self.result else 'with Errors',
                       study = self.study['name'],
                       calculation = self.calculation['name']))
        if self.result:
            logger.info('\n'.join(get_code_aster_error(self.infofile)))

    def init(self):
        self._prepare_paths()
        self._copy_files()
        self._copy_additional_inputfiles()
        self._create_resultfiles()
        self._create_runpy()
        self._create_runsh()

    def run(self):
        if self.config.prepare:
            self.init()
            logger.info('cd to "{0}" and run "run.sh" '.format(buildpath))
        else:
            self.init()
            self._run_bashed()
            self._copy_results()
            return self.response

    def _copy_results(self):
        # try to copy results even if errors occured
        for name,fpath in self.resultfiles.itmes():
            for f in glob.glob(os.path.join(self.buildpath,fpath)):
                outname = name.format(name=os.path.splitext(os.path.split(f)[-1]))
                if os.path.getsize(f) == 0:
                    logger.warn('result file "{0}" is empty'.format(outname))
            self._copy_result(f,os.path.join(self.outputpath,outname))

        # copy the standard result files
        self._copy_result(
            os.path.join(self.buildpath,'fort.6'),
            os.path.join(self.outputpath,self.calculation['name']+'.mess')
        )
        # copy the commandfile as well as the parameters and the mesh
        self._copy_result(
            os.path.join(self.buildpath,'fort.1'),
            os.path.join(self.outputpath,self.calculation['commandfile']),
        )
        self._copy_result(
            os.path.join(buildpath,'fort.20'),
            os.path.join(self.outputpath,self.config.meshfile),
        )
        if self.config.distributionfile:
            self._copy_result(
                self.config.distributionfile,
                os.path.join(self.outputpath,
                             os.path.split(self.config.distributionfile)[-1])
            )
        # copy the zipped base
        self._copy_result(
            os.path.join(self.buildpath,'glob.1'),
            os.path.join(self.outputpath,'glob.1.zip'),zipped=True
        )
        self._copy_result(
            os.path.join(self.buildpath,'pick.1'),
            os.path.join(self.outputpath,'pick.1.zip'),zipped=True
        )
        self.response['glob.1.zip'] = os.path.join(self.outputpath,'glob.1.zip')
        self.response['pick.1.zip'] = os.path.join(self.outputpath,'pick.1.zip')

    def _copy_result(self,fromfile,tofile,zipped=False):
        try:
            if not zipped:
                shutil.copyfile(fromfile,tofile)
            else:
                zipf = zipfile.ZipFile(tofile,'w',allowZip64=True)
                zipf.write(fromfile)
                zipf.close()
        except OSError as e:
            if not self.result:
                raise e
            else:
                logger.warn('ignore exception for copiing result "{0}"'.format(name))


def shutdown(signum,frame):
        stderr = sys.stderr
        stdout = sys.stdout
        sys.stderr = os.devnull
        sys.stdout = os.devnull
        for pname,p in processes:
            if p.is_alive():
                p.terminate()
        sys.stderr = stderr
        sys.stdout = stdout
        print >> sys.stdout, ('killed all calculations trough user')

def main(argv=None):
    signal.signal(signal.SIGINT,shutdown)

    if not argv:
        argv = sys.argv[1:]
    parser = make_pasrer()
    options = parser.parse_args(argv)
    asterclient = AsterClient(options)
    asterclient.run()

if '__main__' == __name__:
    main()
