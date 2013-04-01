import os
import sys
import zipfile
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

class AsterClientException(Exception):
    pass

logger = logging.getLogger('asterclient')
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel('INFO')
def abspath(path,basepath=None):
    if os.path.isabs(path):
        return path
    elif basepath:
        return os.path.abspath(os.path.join(basepath,path))
    else:
        return os.path.abspath(path)

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

def runstudy(calculations,builddir,studyname,studynumber,
        profile,outdir,distributionfile,srcdir,options,
        foreground=False):
    for calculation in calculations:
        buildpath = os.path.join(builddir,
                profile.get('project',''.join(random.choice(
                    string.ascii_uppercase + string.digits)
                    for x in range(5))),studyname,calculation['name'])
        try:
            os.makedirs(buildpath)
        except:
            if options.force:
                logger.info('buildpath is "{0}"'.format(buildpath))
                for x in os.listdir(buildpath):
                    os.remove(os.path.join(buildpath,x))

        # copy the elements
        shutil.copyfile(abspath(profile['elements'],basepath=srcdir),
                os.path.join(buildpath,'elem.1'))
        # copy the mesh file
        shutil.copyfile(abspath(profile['meshfile'],basepath=srcdir),
                os.path.join(buildpath,'fort.20'))

        # create output directory
        outputpath = os.path.join(outdir,studyname,calculation['name'])
        try:
            os.makedirs(outputpath)
        except:
            if not options.force:
                print >> sys.stderr, ('output directory "{0}" exists already,'
                'use the --force option to overwrite it anyways'
                .format(outdir))
                sys.exit(1)
            else:
                # if force option clean up directory
                shutil.rmtree(outputpath)
                os.makedirs(outputpath)

        # copy commandfile to the buildpath
        shutil.copyfile(abspath(calculation['commandfile'],basepath=srcdir),
                os.path.join(buildpath,'fort.1'))
        # copy distributionfile to the buildpath
        if distributionfile:
            shutil.copyfile(abspath(profile['distributionfile']+'.py',basepath=srcdir),
                    os.path.join(buildpath,'distr.py'))
        # if calculation is a continued one copy the results from the
        # previous step
        if 'poursuite' in calculation:
            zipf = zipfile.ZipFile(abspath('glob.1.zip',basepath=os.path.join(outdir,studyname,
                calculation['poursuite'])),'r')
            zipf.extractall(path=buildpath)
            zipf = zipfile.ZipFile(abspath('pick.1.zip',basepath=os.path.join(outdir,studyname,
                calculation['poursuite'])),'r')
            zipf.extractall(path=buildpath)

        # create a list of files which need to be copied to the
        # resultdirectory
        resultfiles = []
        # create additional resultfiles
        if 'resultfiles' in calculation:
            for file_ in calculation['resultfiles']:
                for key in file_:
                    # result files for acces through fortran
                    if type(file_[key]) == int:
                        with open(os.path.join(buildpath,'fort.%s' % file_[key]),'w') as f:
                            resultfiles.append(('fort.%s' % file_[key],key))
                    else:
                        resultfiles.append(('{0}{1}'.format(key,file_[key]),key))

        # copy additional inputfiles
        if 'inputfiles' in calculation:
            for file_ in calculation['inputfiles']:
                try:
                    addsource = os.path.join(srcdir,file_)
                    shutil.copyfile(addsource,os.path.join(buildpath,os.path.split(file_)[-1]))
                    logger.info('copied file "{0}" to "{1}"'.format(addsource,os.path.join(buildpath,os.path.split(file_)[-1])))
                except:
                    logger.exception('failed to copy input file "{0}"'.format(file_))
                    raise

        # create command
        arguments = ['supervisor',
                '--bibpyt', profile['bibpyt'],
                '--commandes','fort.1',
                '--mode',profile['mode'],
                '--rep_outils',profile['rep_outils'],
                '--rep_mat',profile['rep_mat'],
                '--rep_dex',profile['rep_dex'],
                '--bibpyt',profile['bibpyt'],
                '--tpmax',str(profile['tpmax']),
                '--memjeveux',str(profile['memjeveux'])]
        if profile['memory']:
            arguments.extend(['--memory',profile['memory']])
        if profile['max_base']:
            arguments.extend(['--max_base',profile['max_base']])
        if profile['suivi_batch']:
            arguments.append('--suivi_batch')
        try:
            curdir = os.curdir
            os.chdir(buildpath)
            ## add the workingdir to the LD_LIBRARY_PATH
            os.environ['LD_LIBRARY_PATH'] = os.pathsep + buildpath + os.environ['LD_LIBRARY_PATH']
            # execute the shit of it
            #ier = supervisor.main(coreopts=getargs(arguments),params={'params':study[1]})
            # unfortunately POURSUITE only works in a new process,
            # therefore let us call everything in a knew process
            c = ['import sys','sys.path.insert(0,\'{0}\')'.format(profile['bibpyt']),
                    'from Execution.E_SUPERV import SUPERV',
                    'from Execution.E_Core import getargs',
                    'supervisor=SUPERV()',
                    'from distr import parameters']
            if not distributionfile:
                c.append('parqams = {}')
            else:
                c.append('params={{\'params\':parameters[{0}]}};params[\'params\'][\'studynumber\']={0}'
                .format(studynumber))

            c.append('res=supervisor.main(coreopts=getargs({0}),params=params);sys.exit(res)'.format(arguments))
            if foreground:
                tee = '| tee {0}; exit $PIPESTATUS'.format(os.path.join(buildpath,'fort.6'))
                #bashscript = profile['aster'] + '<< END ' + tee + '\n' + ';'.join(c) + '\nEND'
                #bashscript = 'source /data/devel/python/aster/bin/set_env\n' + profile['aster'] + '<< END ' + tee + '\n' + ';'.join(c) + '\nEND'
                #res = subprocess.call([profile['aster'],'-c',';'.join(c)])
                #res = subprocess.call([profile['aster'],'-c',';'.join(c),tee])
                #res = subprocess.call(['bash','-c',bashscript])
            else:
                tee = ('| tee {0} >> {1} ; exit $PIPESTATUS'.
                format(os.path.join(buildpath,'fort.6'),
                        os.path.join(buildpath,'../progress.txt')))
                #res = subprocess.call([profile['aster'],'-c',';'.join(c)],stdout=protocol)
            bashscript = profile['aster'] + '<< END ' + tee + '\n' + ';'.join(c) + '\nEND'
            res = subprocess.call(['bash','-c',bashscript])
            print >> sys.stdout, ('code aster run "{1}:{2}" ended: {0}'.
                    format(termcolor.colored('OK',color='green') if not res else ('with ' +
                        termcolor.colored('Errors',color='red')),
                        studyname,calculation['name']))
            try:
                # copy the results
                for x in resultfiles:
                    # everything else, with globbing
                    allres = glob.glob(x[0])
                    if len(allres) == 0:
                        logger.warn('no files found for "{0}"'.format(x[0]))
                    for f in allres:
                        if os.path.getsize(f) == 0:
                            logger.warn('result file "{0}" is empty'.format(f))
                        else:
                            shutil.copyfile(f,os.path.join(outputpath,f))

                # copy the standard result files
                shutil.copyfile(os.path.join(buildpath,'fort.6'),os.path.join(outputpath,calculation['name']+'.mess'))
                # copy the commandfile as well as the parameters and the mesh
                shutil.copyfile(os.path.join(buildpath,'fort.1'),os.path.join(outputpath,calculation['commandfile']))
                shutil.copyfile(os.path.join(buildpath,'fort.20'),os.path.join(outputpath,profile['meshfile']))
                if distributionfile:
                    shutil.copyfile(os.path.join(buildpath,'distr.py'),os.path.join(outputpath,profile['distributionfile']+'.py'))
                # copy the zipped base
                zipf = zipfile.ZipFile(os.path.join(outputpath,'glob.1.zip'),'w')
                zipf.write('glob.1')
                zipf.close()
                zipf = zipfile.ZipFile(os.path.join(outputpath,'pick.1.zip'),'w')
                zipf.write('pick.1')
                zipf.close()
            except:
                # only raise if run was succesful
                if not res:
                    raise
                else:
                    logger.exception('ignore exception, since code aster run ended with an error')
        except:
            raise
        finally:
            os.chdir(curdir)
        # copy the results

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
    # get default profile
    try:
        profile = yaml.load(pkgutil.get_data(__name__,
                os.path.join('data','defaults.yml')))
    except:
        print >> sys.stderr, ('an error occured, make sure the'
        ' ASTER_ROOT environment variable is set')
        sys.exit(1)
    # read profile
    try:
        profile.update(yaml.load(options.profile.read()))
    except Exception as e:
        print >> sys.stderr, ('following error occured while parsing'
        ' "{0}:\n\n {1}\n please check your profile if it is valid yaml,'.
        format(options.profile.name,e))
        sys.exit(1)
    # make paths absolute
    for pathkey in ['bibpyt','cata','elements','rep_mat','rep_dex','aster']:
        if not os.path.isabs(profile[pathkey]):
            profile[pathkey] = os.path.join(profile['aster_root'],profile['version'],profile[pathkey])
    if not os.path.isabs(profile['rep_outils']):
        profile['rep_outils'] = os.path.join(profile['aster_root'],profile['rep_outils'])
    # import distribution file if given
    if 'distributionfile' in profile:
        if not os.path.isabs(profile['distributionfile']):
            distributionfile = abspath(profile['distributionfile'],basepath=profile['srcdir'])
        try:
            sys.path.append(os.path.split(distributionfile)[0])
            # needs to have the ending py to work
            studies = __import__(os.path.split(distributionfile)[1]).parameters
            sys.path.remove(os.path.split(distributionfile)[0])
            #studies = imp.load_source(*os.path.split(distributionfile)).parameters
        except:
            print >> sys.stderr, 'couldn\'t import the distributionfile, the traceback:'
            raise
    else:
        studies = [{'name':'main'}]
        distributionfile = None
    # do whatever has to be done
    if options.action == 'info':
        if options.studies and studies:
            info_studies(studies)
        elif options.calculations:
            info_calculations(options.profile.name,profile)
        else:
            if studies:
                info_studies(studies)
            info_calculations(options.profile.name,profile)
    elif options.action == 'run':
    # update profile with options from commandline
        if options.workdir:
            profile['workdir'] = options.workdir
        # populate the commandline options into the profile
        for key in ['bibpyt','memjeveux','memory','tpmax','max_base',
                'dbgjeveux','mode','interact','rep_outils','rep_mat',
                'rep_dex','suivi_batch','verif']:
            if getattr(options,key) != None:
                profile[key] = getattr(options,key)
        # check the profile file
        # check if all minimum needed keys are available
        if not 'meshfile' in profile:
            logger.error('you need to specify a meshfile')
            raise AsterClientException

        if not 'calculations' in profile:
            logger.error('you need to specify at least one calculation')
            raise AsterClientException

        # test calculations specified in the profile
        calcnames = []
        for calc in profile['calculations']:
            if not 'name' in calc or not calc['name']:
                logger.error('you need to specify a'
                ' name for every calculation')
                raise AsterClientException
            if calc['name'] in calcnames:
                logger.error('calculation names must be unique, "{0}" isn\'t'.
                        format(calc['name']))
                raise AsterClientException
            else:
                calcnames.append(calc['name'])

            if not 'commandfile' in calc or not calc['commandfile']:
                logger.error('you need to specify a'
                ' commandfile for every calculation')
                raise AsterClientException

        # test the studies specified in the distributionfile
        studynames = []
        studyparams = set(studies[0].keys())
        for study in studies:
            if not 'name' in study:
                logger.error('every study need a name')
                raise AsterClientException
            if study['name'] in studynames:
                logger.error('studynames need to be unique, "{0}" isn\'t'.
                        format(study['name']))
                raise AsterClientException
            else:
                studynames.append(study['name'])

            if set(study.keys()) != studyparams:
                logger.error('all studies need to have the same keys:'
                '\n{0}\nare different'.format(studyparams.difference(
                    set(study.keys()))))
                raise AsterClientException
        # check if all keys have values
        def check_values(dict_):
            for key in dict_:
                if type(dict_[key]) == dict:
                    check_values(dict_[key])
                # test if value is empty, the following list is allowed to be empty
                elif dict_[key] == None and key not in ['memory','max_base']:
                    print >> sys.stderr, ('"{0}" is not valid for'
                    ' "{1}" in your profile'.format(dict_[key],key))
                    return False
            return True
        if not check_values(profile):
            sys.exit(1)


        # get the calculations which should be run
        calculations = []
        if options.calculation:
            calcnames = [i['name'] for i in profile['calculations']]
            for x in options.calculation:
                try:
                    calculations.append(profile['calculations'][int(x)])
                except:
                    try:
                        calculations.append(profile['calculations'][calcnames.index(x)])
                    except:
                        print >> sys.stderr, ('invalid choice "{0}"'
                        ' for calculation'.format(x))
        else:
            # take all if none is specified
            calculations = profile['calculations']
        # get the studies which should be run
        studies_to_run = []
        if options.study:
            for x in options.study:
                try:
                    studies_to_run.append((int(x),studies[int(x)]))
                except:
                    try:
                        studies_to_run.append((studynames.index(x),
                            studies[studynames.index(x)]))
                    except:
                        print >> sys.stderr, ('invalid choice "{0}"'
                        ' for study'.format(x))
        else:
            # take all if none is specified
            studies_to_run = [ (i,studies[i]) for i in range(len(studies))]
        # set up the paths
        srcdir = os.path.abspath(profile.get('srcdir','.'))
        builddir = profile.get('workdir',None)
        outdir = os.path.abspath(profile.get('outdir','results'))
        # setup the builddir
        if not builddir:
            builddir = tempfile.mkdtemp('_asterclient',
                    profile.get('project','tmp'))
            atexit.register(shutil.rmtree,builddir)
        else:
            builddir = os.path.join(abspath(builddir))
            try:
                os.makedirs(builddir)
            except:
                if not options.force:
                    print >> sys.stderr, ('work directory "{0}" exists already,'
                    'use the --force option to overwrite it anyways'
                    .format(builddir))
                    sys.exit(1)
                else:
                    # if force option clean up directory
                    pass


        # run the shit
        sys.path.append(profile['bibpyt'])
        # symlink cata to Cata/cata
        try:
            os.symlink(os.path.join(profile['cata'],'cata.py'),os.path.join(profile['bibpyt'],'Cata','cata.py'))
        except OSError as e:
            if e.errno == os.errno.EEXIST:
                os.remove(os.path.join(profile['bibpyt'],'Cata','cata.py'))
                os.symlink(os.path.join(profile['cata'],'cata.py'),os.path.join(profile['bibpyt'],'Cata','cata.py'))

        from Execution.E_SUPERV import SUPERV
        from Execution.E_Core import getargs
        supervisor = SUPERV()

        processes = []

        if options.sequential:
            for studynumber,study in studies_to_run:
                runstudy(**{'calculations':calculations,
                        'builddir':builddir,'studyname':study['name'],'studynumber':studynumber,
                        'profile':profile,'outdir':outdir,'srcdir':srcdir,
                        'distributionfile':distributionfile,
                        'options':options,'foreground':True})

        else:
            for studynumber,study in studies_to_run:
                processes.append((study,multiprocessing.Process(target=runstudy,
                    kwargs={'calculations':calculations,
                    'builddir':builddir,'studyname':study['name'],'studynumber':studynumber,
                    'profile':profile,'outdir':outdir,'srcdir':srcdir,
                    'distributionfile':distributionfile,
                    'options':options})))
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

if '__main__' == __name__:
    main()
