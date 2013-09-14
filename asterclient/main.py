import pickle
import os
import sys
import zipfile
import shutil
import copy
import time
import glob
import imp
import tempfile
import argparse
import subprocess
import signal
import multiprocessing
import threading
import logging
import logging.handlers
import logutils.queue
import configreader
from . import translator
from pkg_resources import resource_stream

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

def get_code_aster_error(filename):
    res = []
    record = False
    with open(filename,'r',errors='ignore') as f:
        for line in f:
            if line.startswith('>>') and not record:
                record = True
            elif line.startswith('>>') and record:
                record = False
            elif record and line.strip():
                res.append(line.replace('!',''))
    return res

class Consumer(multiprocessing.Process):

    def __init__(self, task_queue, kill_event,counter,
                 counter_lock,num_consumers,**kwargs):
        # let us ignore interupts
        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue
        self.kill_event = kill_event
        self.counter = counter
        self.counter_lock = counter_lock
        self.num_consumers = num_consumers
        self.kwargs = kwargs

    def run(self):
        proc_name = self.name
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        while not self.kill_event.is_set():
            next_task = self.task_queue.get()
            if not next_task:
                # None as death pill
                break
            answer = next_task(
                self.kill_event,**self.kwargs)
            if not self.kill_event.is_set():
                for calc in next_task.run_after:
                    calc.unset_logger()
                    self.task_queue.put(calc)
            with self.counter_lock:
                self.counter.value -= 1
                if self.counter.value == 0 or self.kill_event.is_set():
                    # now we enque the death pills for each consumer
                    for i in range(self.num_consumers):
                        self.task_queue.put(None)

class QueueListener(logutils.queue.QueueListener):
    def handle(self, record):
        for handler in self.handlers:
            if record.levelno < handler.level:
                continue
            handler.handle(record)

class AsterClientException(Exception):
    pass

class Parser(object):
    def __init__(self,argv):
        self.argv = argv
        self._preparser = None
        self._preoptions = None
        self._options = None
        self._defaultprofile = None
        self._profile = None
        self._asterparser = None
        self._clientparser = None
        self._parser = None

    @property
    def preparser(self):
        if not self._preparser:
            # parser for the global options
            preparser = argparse.ArgumentParser(add_help=False)
            preparser.add_argument('--log-level',default='INFO',
                choices=['DEBUG', 'INFO', 'WARN','ERROR'],
                    help='specify the logging level')
            preparser.add_argument('--logfile',
                    help='path of the used logfile')
            preparser.add_argument('-p','--profile',
                    help='specify profile file')
            self._preparser = preparser
        return self._preparser

    @property
    def asterparser(self):
        if not self._asterparser:
            asterparser = argparse.ArgumentParser(add_help=False)
            asterparser.add_argument('--bibpyt',
                help="path to Code_Aster python source files")
            asterparser.add_argument('--memjeveux',type=int,
                help="maximum size of the memory taken by the execution (in Mw)")
            asterparser.add_argument('--memory',type=int,
                help="maximum size of the memory taken by the execution (in MB)")
            asterparser.add_argument('--tpmax',type=int,
                help="limit of the time of the execution (in seconds)")
            asterparser.add_argument('--max_base',type=int,
                help="limit of the size of the results database")
            asterparser.add_argument('--dbgjeveux',action='store_true',
                help="maximum size of the memory taken by the execution in Mw")
            asterparser.add_argument('--mode',type=str,
                help="execution mode (interactive or batch)")
            asterparser.add_argument('--interact', action='store_true', default=False,
                help="as 'python -i' works, it allows to enter commands after the "
                        "execution of the command file.")
            asterparser.add_argument('--rep_outils',
                help="directory of Code_Aster tools (ex. $ASTER_ROOT/outils)")
            asterparser.add_argument('--rep_mat',
                help="directory of materials properties")
            asterparser.add_argument('--rep_dex',
                help="directory of external datas (geometrical datas or properties...)")
            asterparser.add_argument('--suivi_batch',action='store_true',default=True,
                help="force to flush of the output after each line")
            asterparser.add_argument('--verif', action='store_true', default=False,
                help="only check the syntax of the command file is done")
            # set defaults from profile
            self._asterparser = asterparser
        return self._asterparser

    @property
    def clientparser(self):
        if not self._clientparser:
            clientparser = argparse.ArgumentParser(add_help=False)
            clientparser.add_argument('-s','--study',nargs='*',
                    help='run the these studies')
            clientparser.add_argument('-c','--calculation',nargs='*',
                    help='run the these calculations')
            clientparser.add_argument('--clean',action='store_true',
                    help='cleans existing workdir and resultdir')
            clientparser.set_defaults(**self.profile)
            self._clientparser = clientparser
        return self._clientparser

    @property
    def parser(self):
        if not self._parser:
            parser = argparse.ArgumentParser(parents=[self.preparser])
            subparsers = parser.add_subparsers(dest='action')
            info = subparsers.add_parser('info',
                parents=[self.preparser,self.asterparser,self.clientparser])
            info.set_defaults(**self.profile)
            interactive = subparsers.add_parser('interactive',
                parents=[self.preparser,self.asterparser,self.clientparser])
            interactive.set_defaults(**self.profile)
            run = subparsers.add_parser('run',
                parents=[self.preparser,self.asterparser,self.clientparser])
            run.add_argument('--parallel',action='store_true',
                    help='don\'t dispatch but run parallel')
            run.add_argument('--max-parallel',required=False,default=10,
                    help='limit the number of parallel processes',type=int)
            run.add_argument('--hide-aster',action='store_true',
                    help='hide the output of code aster')
            run.add_argument('--dispatch',action='store_true',
                    help='start the calculation but doesn\'t wait for it to finnish')
            run.set_defaults(**self.profile)
            copyresult = subparsers.add_parser('copyresult',
                parents=[self.preparser,self.clientparser],
                help='copies the results to the result folder, useful if'
                    'calculation was dispatched or run by hand')
            copyresult.set_defaults(**self.profile)
            prepare = subparsers.add_parser('prepare',
                parents=[self.preparser,self.clientparser],
                help="only prepare everything")
            prepare.set_defaults(**self.profile)
            self._parser = parser
        return self._parser

    @property
    def options(self):
        if not self._options:
            self._options = vars(self.parser.parse_args(self.argv))
        return self._options

    @property
    def preoptions(self):
        if not self._preoptions:
            self._preoptions = self.preparser.parse_known_args(self.argv)[0]
        return self._preoptions

    @property
    def defaultprofile(self):
        if not self._defaultprofile:
            with resource_stream(__name__, 'data/default.conf') as f:
                self._defaultprofile = configreader.Config(
                    f,namespace={'os.getenv':os.getenv})
        return self._defaultprofile

    @property
    def profile(self):
        if not self._profile:
            profile = copy.copy(self.defaultprofile)
            try:
                with open(self.preoptions.profile,'r') as f:
                    profile.update(configreader.Config(
                        f,namespace={'os.getenv':os.getenv}))
            except Exception as e:
                raise AsterClientException(
                    'the profile {0} couldn\'t be parsed:\n\n\t{1}'
                    .format(self.preoptions.profile,e))
            self._profile = profile
        return self._profile

class AsterClient(object):
    def __init__(self,options,logger=None):
        self.options = options
        self._basepath = None
        self.logger = logger
        self._set_logger()
        self._studies_to_run = None
        self._calculations_to_run = None
        self._executions_nested = None
        self._executions_flat = None
        self._num_executions = None
        self._distributionfile = -1
        self._remove_at_exit = []
        self._studies = None
        self._calculationslist = None
        self._calculationsdict = None
        self._kill_event = multiprocessing.Event()
        signal.signal(signal.SIGINT,self.shutdown)
        self._manager = multiprocessing.Manager()
        self._log_queue  = self._manager.Queue(-1)
        self._listener = QueueListener(self._log_queue, *self.logger.handlers)
        # start the log listener
        self._listener.start()

    def _absolutize_option_paths(self):
        for pathkey in ['bibpyt','cata','elements','rep_mat','rep_dex','aster']:
            self.options[pathkey] = os.path.join(self.options['aster_root'],
                self.options['version'],self.options[pathkey])
        if not os.path.isabs(self.options['rep_outils']):
            self.options['rep_outils'] = os.path.join(
                self.options['aster_root'],self.options['rep_outils'])
        self.options['outdir'] = self._abspath(self.options['outdir'])

    @property
    def basepath(self):
        if not self._basepath:
            self._basepath = os.path.abspath(self.options.get('srcdir','.'))
        return self._basepath

    def _set_logger(self):
        if not self.logger:
            logger = logging.getLogger('asterclient')
            console_handler = logging.StreamHandler(sys.stdout)
            console_formatter = logging.Formatter(
            '%(processName)-10s %(name)s %(levelname)-8s %(message)s')
            console_handler.setFormatter(console_formatter)
            if self.options["logfile"]:
                file_handler = logging.FileHandler(self.options["logfile"], 'a')
                file_formatter = logging.Formatter(
                    '%(asctime)s %(processName)-10s %(name)s %(levelname)-8s %(message)s')
                file_handler.setFormatter(file_formatter)
            else:
                file_handler = None
            logger.setLevel(self.options["log_level"])
            console_handler.setLevel(self.options["log_level"])
            if file_handler:
                file_handler.setLevel(self.options["log_level"])
            if file_handler:
                logger.addHandler(file_handler)
            logger.addHandler(console_handler)
            self.logger = logger

    def _abspath(self,path):
        if os.path.isabs(path):
            return path
        else:
            return os.path.abspath(os.path.join(self.basepath,path))

    @property
    def calculationsdict(self):
        if not self._calculationsdict:
            self._calculationsdict = {c['name']:c for c in self.calculationslist}
        return self._calculationsdict

    @property
    def calculationslist(self):
        if not self._calculationslist:
            self._calculationslist = self._load_calculations()
        return self._calculationslist

    def _load_calculations(self):
        # test calculations specified in the profile
        if not 'calculations' in self.options:
            raise AsterClient('there are no calcualtions specified')
        calcs = []
        for i,calc in enumerate(self.options['calculations']):
            name = calc.get('name','')
            if not name:
                raise AsterClientException('no calcualtion name specified')
            if name in calcs:
                raise AsterClientException('calculation name {0} isn\'t unique'
                                        .format(name))

            commandfile = calc.get('commandfile','')
            if not commandfile:
                raise AsterClientException('no commandfile specified for "{0}"'
                                       .format(name))
            calc['commandfile'] = self._abspath(commandfile)
            if 'inputfiles' in calc:
                inputfiles = []
                for i,fpath in enumerate(calc['inputfiles']):
                    inputfiles.append(self._abspath(fpath))
                calc['inputfiles'] = inputfiles
            calcs.append(calc)
        return calcs

    @property
    def distributionfile(self):
        if self._distributionfile == -1:
            if 'distributionfile' in self.options:
                distr = self._abspath(self.options["distributionfile"])
            else:
                studies = [{'name':'main'}]
                distr = None
            self._distributionfile = distr
            self.options["distributionfile"] = distr
        return self._distributionfile

    @property
    def studies(self):
        if not self._studies:
            self._studies = self._load_studies()
        return self._studies

    def _load_studies_from_distr(self):
        name = os.path.splitext(
                os.path.basename(self.distributionfile))[0]
        try:
            studies = imp.load_source(
                    name,self.distributionfile).parameters
        except Exception as e:
            raise AsterClientException(
                    'couldn\'t import distributionfile, %s'%e)
        study_keys = []
        study_names = []
        for i,study in enumerate(studies):
            study_keys.append(tuple(study.keys()))
            # set the number
            study['number'] = i
            # test the studies specified in the distributionfile
            name = study.get('name','')
            if not name:
                raise AsterClientException('every study needs a name')
            if name in study_names:
                raise AsterClientException('study names {0} isn\'t unique'.format(name))
            else:
                study_names.append(name)

            meshfile = study.get('meshfile',self.options.get('meshfile'))
            if not meshfile:
                raise AsterClientException('no meshfile specified for "{0}"'.format(name))
            study['meshfile'] = self._abspath(meshfile)
        # check if all studies have the same keys
        for i,study in enumerate(study_keys[1:]):
            if study != study_keys[i-1]:
                raise AsterClientException(
                    'all study "{0}" has different keys'.format(i))
        return studies

    def _load_studies(self):
        if self.distributionfile:
            return self._load_studies_from_distr()
        else:
            # we need to get the relevant data and create a study
            study = {'name':'','number':0}
            meshfile = self.options.get('meshfile')
            if not meshfile:
                raise AsterClientException('no meshfile specified')
            study['meshfile'] = self._abspath(meshfile)
        return [study]

    @property
    def calculations_to_run(self):
        # get the calculations which should be run
        if not self._calculations_to_run:
            if self.options["calculation"]:
                calculations = []
                for x in self.options["calculation"]:
                    try:
                        calculations.append(self.calculationslist[int(x)])
                    except:
                        try:
                            calculations.append(self.calculationsdict[x])
                        except:
                            raise AsterClientException(
                                    'there is no calculation "{0}"'.format(x))
            else:
                # take all if none is specified
                calculations = self.calculationslist
            self._calculations_to_run = calculations
        return self._calculations_to_run

    @property
    def studies_to_run(self):
        # get the studies which should be run
        if not self._studies_to_run:
            if self.options["study"]:
                studies = []
                studynames = [study['name'] for study in self.studies]
                for x in self.options["study"]:
                    try:
                        # get by index
                        key = int(x)
                    except:
                        key = None
                    if key == None:
                        try:
                            key = studynames.index(x)
                        except:
                            raise AsterClientException(
                                    'ther is no study "{0}"'.format(x))
                    studies.append(self.studies[key])
            else:
                studies = self.studies
            self._studies_to_run = studies
        return self._studies_to_run

    @property
    def executions_nested(self):
        if not self._executions_nested:
            self._create_executions()
        return self._executions_nested

    @property
    def executions_flat(self):
        if not self._executions_flat:
            self._create_executions()
        return self._executions_flat

    @property
    def num_executions(self):
        if not self._num_executions:
            self._create_executions()
        return self._num_executions

    def _ensure_workdir(self):
        # setup the workdir
        workdir = self.options.get('workdir')
        if not workdir:
            workdir = tempfile.mkdtemp(
                suffix='.asterclient',prefix=self.options.get('project'))
            self._remove_at_exit.append(workdir)
            self.logger.info('created temporary dir "{0}"'.format(workdir))
        else:
            workdir = self._abspath(workdir)
        self.options['workdir'] = workdir

    def _create_executions(self):
        self._absolutize_option_paths()
        self._ensure_workdir()
        executions = []
        executions_flat = []
        if self.studies_to_run:
            for study in self.studies_to_run:
                tmp = {}
                for calc in self.calculations_to_run:
                    need = calc.get('poursuite')
                    tmp[calc['name']] = Calculation(
                        self.options,study,calc,need,self._log_queue)
                for calc in tmp.values():
                    if calc.needs and calc.needs in tmp:
                        tmp[calc.needs].append_run_after(calc)
                    else:
                        executions.append(calc)
                    executions_flat.append(calc)

        self._executions_nested = executions
        self._executions_flat = executions_flat
        self._num_executions = len(executions_flat)

    def run_parallel(self,**kwargs):
        task_queue = multiprocessing.Queue()
        counter_lock = multiprocessing.Lock()
        counter = multiprocessing.Value('i',(self.num_executions))
        num_consumers = min((self.options.get("max_parallel",10),
                             self.num_executions))
        # Enqueue jobs
        for calc in self.executions_nested:
            # remove logger object if already hase one
            calc.unset_logger()
            task_queue.put(calc)
        # Start consumers
        consumers = []
        for i in range(num_consumers):
            c = Consumer(task_queue,self._kill_event,counter,
                         counter_lock,num_consumers,**kwargs)
            consumers.append(c)
            c.start()
        # Wait for all of the tasks to finish
        for c in consumers:
            c.join()
        # remove tmp dir again
        if not self.options.get("dispatch") and self.options.get("clean"):
            self.remove_tmp()

    def run_sequential(self,**kwargs):
        def run_sequential_recursive(calcs):
            for calc in calcs:
                if self._kill_event.is_set():
                    break
                calc(kill_event=self._kill_event,**kwargs)
                run_sequential_recursive(calc.run_after)
        run_sequential_recursive(self.executions_nested)
        if not self.options.get("dispatch") and self.options.get("clean"):
            self.remove_tmp()

    def shutdown(self,sig,frame):
        self.logger.warn('will shutdown all processes, interrupt')
        self._kill_event.set()

    def prepare(self,**kwargs):
        for calc in self.executions_nested:
            # we only need to run the toplevel executions here since the
            # others would fail anyways, or wouldn't be up to date
            # maybe still useful, don't know how would be ideal
            calc.prepare(**kwargs)
    def copy_results(self,**kwargs):
        for calc in self.executions_flat:
            self.logger.debug('starting to copy results for "{0}"'
                              .format(calc.name))
            calc.copy_results(**kwargs)

    def remove_tmp(self):
        for f in self._remove_at_exit:
            shutil.rmtree(f)
            self.logger.info('removing temporary object "{0}"'.format(f))

    def info(self):
        self._info_studies()
        self._info_calculations()
    def _info_studies(self):
        """ print the available studies
        of the given profile file"""
        print('\navailable parametric studies:\n')
        i = 0
        for key in self.studies:
            print('\t{0}: {1}\n'.format(i,key['name']))
            i +=1

    def _info_calculations(self):
        """ print the available calculations
        of the given profile file"""
        print('\navailable calculations:\n')
        for i,calc in enumerate(self.options['calculations']):
            print('\t{0}: {1}\n'.format(i,calc['name']))

class Calculation(object):
    # unfortunately POURSUITE only works in a new process,
    # therefore let us call everything in a knew process
    def __init__(self,config,study,calculation,needs,queue):
        self.config = config
        self.study = study
        self.calculation = calculation
        # my needs calculation, needed if we resume
        self.needs = needs
        self.name = '{0}:{1}'.format(study.get('name'),calculation['name'])
        self._initiated = False
        self._processing = False
        self.success = False
        self.finnished = False
        self._run_after = []
        self._killed = None
        self._resultfiles = None
        self._logger = None
        self._queue = queue

    @property
    def buildpath(self):
        return os.path.join(self.config["workdir"],
                            self.study.get('name'),self.calculation['name'])

    @property
    def outputpath(self):
        return os.path.join(self.config["outdir"],
                            self.study.get('name'),self.calculation['name'])

    @property
    def infofile(self):
        return os.path.join(self.buildpath,'fort.6')

    def __str__(self):
        return '<Calculation: %s>'%self.name

    @property
    def run_after(self):
        return self._run_after

    def append_run_after(self,calc):
        self._run_after.append(calc)

    def _clean_path(self,path):
        try:
            shutil.rmtree(path)
            os.makedirs(path)
            self.logger.debug('cleaned path "{0}"'.format(path))
        except:
            self.logger.error('failed to clean path "{0}"'.format(path))

    def _prepare_buildpath(self):
        # make sure buildpath exists and is clean
        try:
            os.makedirs(self.buildpath)
        except:
            if self.config.get("clean"):
                self._clean_path(self.buildpath)
            else:
                self.logger.warn('buildpath "{0}" exists and holds data'
                                 .format(self.buildpath))

    def _prepare_outputpath(self):
        # make sure output directory exists and is clean
        try:
            os.makedirs(self.outputpath)
        except:
            if self.config["clean"]:
                self._clean_path(self.outputpath)
            else:
                self.logger.warn('outputpath "{0}" exists and holds data'.format(self.outputpath))

    def _copy_files(self):
        # copy the elements catalog
        shutil.copyfile(
            self.config["elements"],os.path.join(self.buildpath,'elem.1'))
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
                self.logger.debug('try to copy glob.1.zip from %s'%glob1)
                with zipfile.ZipFile(glob1,'r',allowZip64=True) as zipf:
                    zipf.extractall(path=self.buildpath)
                self.logger.debug('try to copy pick.1.zip from %s'%pick1)
                with zipfile.ZipFile(pick1,'r',allowZip64=True) as zipf:
                    zipf.extractall(path=self.buildpath)
            except Exception as e:
                raise AsterClientException(
                    'failed to copy glob.1 and pick.1 from "{0}"'.format(self.needs))

    def _createresultfiles(self):
        # the resultfiles need already to be created for fortran
        # create a list of files which need to be copied to the
        # resultdirectory
        if self.resultfiles:
            for key,f in self.calculation['resultfiles'].items():
                # result files for acces through fortran
                if type(f) == int:
                    name = 'fort.%s'%f
                    # touch the file
                    with open(os.path.join(self.buildpath,name),'w') as f:
                        os.utime(f.name, None)

    @property
    def resultfiles(self):
        if self._resultfiles:
            return self._resultfiles
        else:
            resultfiles = {}
            if 'resultfiles' in self.calculation:
                for key,f in self.calculation['resultfiles'].items():
                    # result files for acces through fortran
                    if type(f) == int:
                        name = 'fort.%s'%f
                        resultfiles[key] = name
                    else:
                        resultfiles[key] = f
            self._resultfiles = resultfiles
            return self._resultfiles

    def _copy_additional_inputfiles(self):
        if 'inputfiles' in self.calculation:
            for f in self.calculation['inputfiles']:
                try:
                    shutil.copyfile(f,os.path.join(self.buildpath,os.path.basename(f)))
                except:
                    self.logger.error('failed to copy input file "{0}"'.format(f))

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

        runpy = RUNPY_TEMPLATE.format(aster=self.config["aster"],
                   bibpyt=self.config["bibpyt"],
                   syspath=sys.path,
                   arguments=arguments,
                   params=self.study)

        self.runpy_path = os.path.join(self.buildpath,'run.py')
        with open(self.runpy_path,'w') as f:
            f.write(runpy)
        os.chmod(self.runpy_path,0o777)

    def _create_runsh(self):
        runsh = RUNSH_TEMPLATE.format(LD_LIBRARY_PATH=self.buildpath,runpy=self.runpy_path)

        self.runsh_path = os.path.join(self.buildpath,'run.sh')
        with open(self.runsh_path,'w') as f:
            f.write(runsh)
        os.chmod(self.runsh_path,0o777)

    def _run_bashed(self):
        if not self.config.get("hide_aster"):
            tee = '| tee {0}; exit $PIPESTATUS'.format(self.infofile)
        else:
            tee = '2&> {0};exit $PIPESTATUS'.format(self.infofile)
        bashscript = '{runsh} {tee}'.format(runsh=self.runsh_path,tee=tee)
        self.subprocess = subprocess.Popen(
                ['bash','-c',bashscript],cwd=self.buildpath)
        wait = not self.config.get("dispatch") # if dispatch option set we just go on
        while wait:
            if self.subprocess.poll() != None:
                wait = False
            elif self._kill_event and self._kill_event.is_set():
                self.subprocess.kill()
                wait = False
                self._killed = True
            else:
                time.sleep(2)
        if self.subprocess.returncode == 0:
            self.success = True

    def init(self):
        if not self._initiated: # we don't wanna initialize multiple times
            try:
                self._prepare_buildpath()
                self._copy_files()
                self._copy_additional_inputfiles()
                self._createresultfiles()
                self._create_runpy()
                self._create_runsh()
                self._initiated = True
            except Exception as e:
                self.logger.error('couldn\'t prepare the execution, %s'%e)

    def _run_info(self):
        if self.subprocess.returncode == 0:
            self.logger.info('Code Aster run ended OK')
        elif not self._killed:
            error = '\n'.join(get_code_aster_error(self.infofile))
            error_en = translator.Translator(error,'fr','en').get()
            self.logger.warn('Code Aster run ended with ERRORS:\n\n\t{0}\n'
                             .format('\n\t'.join(error_en)))


    def unset_logger(self):
        self._logger = None

    @property
    def logger(self):
        if not self._logger:
            logger = logging.getLogger(self.name)
            handler = logutils.queue.QueueHandler(self._queue)
            logger.addHandler(handler)
            self._logger = logger
        return self._logger

    def setloglevel(self):
        try:
            self.logger.setLevel(self.config.get('log_level','DEBUG'))
        except:
            self.logger.setLevel('DEBUG')

    def __call__(self,kill_event=None,**kwargs):
        self.config.update(kwargs)
        self.setloglevel()
        self._kill_event = kill_event
        if not self._processing: # we certainly don't wanna run multiple times
            self._processing = True
            self.logger.info('started processing')
            self.init()
            if self._initiated:
                self._run_bashed()
                if not self.config.get("dispatch"):
                    self._run_info()
                    self.copy_results()
                    self._processing = False
                    self.finnished = True
                    self.logger.info('finnished processing')
                else:
                    self.logger.info('dispatched run to process "{0}"'
                            .format(self.subprocess.pid))

    def prepare(self,**kwargs):
        self.config.update(kwargs)
        self.setloglevel()
        self.init()
        self.logger.info('prepared run.sh in "{0}"'.format(self.buildpath))

    def copy_results(self,**kwargs):
        self.config.update(kwargs)
        self.setloglevel()
        self._prepare_outputpath()
        # try to copy results even if errors occured
        for name,fpath in self.resultfiles.items():
            for f in glob.glob(os.path.join(self.buildpath,fpath)):
                outname = os.path.basename(f)
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
            self.config["meshfile"])),
        )
        if self.config["distributionfile"]:
            self._copyresult(
                self.config["distributionfile"],
                os.path.join(self.outputpath,os.path.basename(
                    self.config["distributionfile"]))
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
        self.logger.info('copied results to "{0}"'.format(self.outputpath))

    def _copyresult(self,fromfile,tofile,zipped=False):
        if self.success and os.path.getsize(fromfile) == 0:
            self.logger.warn('result file "{0}" is empty'.format(outname))
        self.logger.debug('copiing "{0}"'.format(fromfile))
        try:
            if not zipped:
                shutil.copyfile(fromfile,tofile)
            else:
                zipf = zipfile.ZipFile(tofile,'w',allowZip64=True)
                zipf.write(fromfile,arcname=os.path.basename(fromfile),
                           compress_type=zipfile.ZIP_DEFLATED)
                zipf.close()
        except Exception as e:
            if self.success:
                raise e
            else:
                self.logger.debug('ignore exception for copying result "{0}"'
                            .format(os.path.basename(fromfile)))

def main(argv=None):
    if not argv:
        argv = sys.argv[1:]

    parser = Parser(argv)
    options = parser.options
    if options["log_level"].upper() == 'DEBUG' and not options.get("parallel"):
        from . import debug
    asterclient = AsterClient(options)
    try:
        if options["action"] == 'info':
            asterclient.info()
        elif options["action"] == 'prepare':
            asterclient.prepare()
        elif options["action"] == 'copyresult':
            asterclient.copy_results()
        elif options["action"] == 'interactive':
            # https://github.com/ipython/ipython/wiki/Cookbook%3a-Updating-code-for-use-with-IPython-0.11-and-later#embedding-api
            import IPython
            from IPython.config.loader import Config
            cfg = Config()
            cfg.TerminalInteractiveShell.banner1 = ''
            cfg.TerminalInteractiveShell.banner2 = '"cl" is your access to the client\n'
            cfg.PromptManager.in_template="asterclient [\\#]> "
            cfg.PromptManager.out_template="asterclient [\\#]: "
            namespace = {'cl':asterclient}
            IPython.embed(config=cfg, user_ns=namespace)
        elif options["action"] == 'run':
            if options["parallel"]:
                asterclient.run_parallel()
            else:
                asterclient.run_sequential()
        # now also close the log listener
        asterclient._listener.stop()
    except KeyboardInterrupt:
        pass
        #logger.error('killed all calculations through interrupt')

if '__main__' == __name__:
    main()
