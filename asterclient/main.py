import os
import re
import sys
import zipfile
import shutil
import ipdb
import yaml
import time
import imp
import atexit
import tempfile
import argparse
import pkgutil
import subprocess
import threading
import multiprocessing


ERROR = re.compile('Exception')

class Tee(object):
    def __init__(self, name, mode):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        sys.stdout = self
    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()
    def close(self):
        self.__del__()
    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

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
    runparser.add_argument('--suivi_batch',
        help="force to flush of the output after each line")
    runparser.add_argument('--verif', action='store_true', default=False,
        help="only check the syntax of the command file is done")
    return parser

def info_studies(parameters,studyname=None):
    """ print the available studies
    of the given profile file"""
    if not studyname:
        print('\navailable parametric studies:\n')
        for i in range(len(parameters)):
            print('\t{0}: {1}\n'.format(i,parameters[i][0]))
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
        profile,outdir,distributionfile,srcdir,options,foreground=False):
    for calculation in calculations:
        buildpath = os.path.join(builddir,studyname,calculation['name'])
        try:
            os.makedirs(buildpath)
        except:
            if options.force:
                for x in os.listdir(buildpath):
                    os.remove(x)

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
            shutil.copyfile(abspath(profile['distributionfile'],basepath=srcdir),
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
                    with open(os.path.join(buildpath,'fort.%s' % file_[key]),'w') as f:
                        resultfiles.append(('fort.%s' % file_[key],key))
        # create command
        arguments = ['supervisor',
                '--bibpyt', profile['bibpyt'],
                '--commandes','fort.1',
                '--mode',profile['mode'],
                '--rep_outils',profile['rep_outils'],
                '--rep_mat',profile['rep_mat'],
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
            # execute the shit of it
            #ier = supervisor.main(coreopts=getargs(arguments),params={'params':study[1]})
            # unfortunately POURSUITE only works in a new process,
            # therefore let us call everything in a knew process
            c = ['import sys','sys.path.append(\'{0}\')'.format(profile['bibpyt']),
                    'from Execution.E_SUPERV import SUPERV',
                    'from Execution.E_Core import getargs',
                    'supervisor=SUPERV()',
                    'from distr import parameters;params={{\'params\':parameters[{0}][1]}}'
                    .format(studynumber) if distributionfile else 'params={}',
                    'res=supervisor.main(coreopts=getargs({0}),params=params);sys.exit(res)'.format(arguments)]
            protocol = open(os.path.join(buildpath,'fort.6'),'w')
            if foreground:
                res = subprocess.call([profile['aster'],'-c',';'.join(c)])
            else:
                res = subprocess.call([profile['aster'],'-c',';'.join(c)],stdout=protocol)
            protocol.close()
            protocol = open(protocol.name,'r')
            protocol_content = protocol.read()
            protocol.close()
            print >> sys.stdout, ('code aster run {1}:{2} ended: {0}'.
                    format('OK' if not res else 'with Errors',
                        studyname,calculation['name']))
            try:
                # copy the results
                for x in resultfiles:
                    shutil.copyfile(os.path.join(buildpath,x[0]),os.path.join(outputpath,x[1]))
                # copy the standard result files
                shutil.copyfile(os.path.join(buildpath,'fort.6'),os.path.join(outputpath,calculation['name']+'.mess'))
                # copy the commandfile as well as the parameters and the mesh
                shutil.copyfile(os.path.join(buildpath,'fort.1'),os.path.join(outputpath,calculation['commandfile']))
                shutil.copyfile(os.path.join(buildpath,'fort.20'),os.path.join(outputpath,profile['meshfile']))
                if distributionfile:
                    shutil.copyfile(distributionfile,os.path.join(outputpath,profile['distributionfile']))
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
        except:
            raise
        finally:
            os.chdir(curdir)
        # copy the results


def main(argv=None):
    if not argv:
        argv = sys.argv[1:]
    parser = make_pasrer()
    options = parser.parse_args(argv)
    # get default profile
    profile = yaml.load(pkgutil.get_data(__name__,
            os.path.join('data','defaults.yml')))
    # read profile
    try:
        profile.update(yaml.load(options.profile.read()))
    except Exception as e:
        print >> sys.stderr, ('following error occured while parsing'
        ' "{0}:\n\n {1}\n please check your profile if it is valid yaml,'.
        format(options.profile.name,e))
        sys.exit(1)
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
    def checkkeyinprofile(profile,key):
        if not key in profile:
            print >> sys.stderr, ('"{0}" is missing in '
            'your profile file'.format(key))
            sys.exit(1)


    checkkeyinprofile(profile,'meshfile')
    checkkeyinprofile(profile,'calculations')
    def check_min_keys(dict_):
        if not 'name' in calc or not calc['name']:
            print >> sys.stderr, ('you need to specify a'
            ' name for every calculation')
            return False
        if not 'commandfile' in calc or not calc['commandfile']:
            print >> sys.stderr, ('you need to specify a'
            ' commandfile for every calculation')
            return False
        return True
    for calc in profile['calculations']:
        if not check_min_keys(calc):
            sys.exit(1)
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

    # import distribution file if given
    if 'distributionfile' in profile:
        if not os.path.isabs(profile['distributionfile']):
            distributionfile = abspath(profile['distributionfile'],basepath=profile['srcdir'])
        try:
            studies = imp.load_source(*os.path.split(distributionfile)).parameters
        except:
            print >> sys.stderr, 'couldn\'t import the distributionfile, the traceback:'
            raise
    else:
        studies = ('',{})
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
            studynames = [i[0] for i in studies]
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
                    shutil.rmtree(builddir)
                    os.makedirs(builddir)

        # run the shit
        sys.path.append(profile['bibpyt'])
        from Execution.E_SUPERV import SUPERV
        from Execution.E_Core import getargs
        supervisor = SUPERV()
        processes = []
        if options.sequential:
            for studynumber,study in studies_to_run:
                runstudy(**{'calculations':calculations,
                        'builddir':builddir,'studyname':study[0],'studynumber':studynumber,
                        'profile':profile,'outdir':outdir,'srcdir':srcdir,
                        'distributionfile':distributionfile,
                        'options':options,'foreground':True})

        else:
            for studynumber,study in studies_to_run:
                processes.append(multiprocessing.Process(target=runstudy,
                    kwargs={'calculations':calculations,
                    'builddir':builddir,'studyname':study[0],'studynumber':studynumber,
                    'profile':profile,'outdir':outdir,'srcdir':srcdir,
                    'distributionfile':distributionfile,
                    'options':options}))
            # start the process stepped
            c = 0
            ncpus = multiprocessing.cpu_count()
            for i in range(len(processes)):
                processes[i].start()
                c += 1
                if c == ncpus:
                    time.sleep(1)
                    for j in range(ncpus):
                        processes[j].join()
                c = 0

if '__main__' == __name__:
    main()
