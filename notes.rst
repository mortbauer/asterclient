Notes
#####
Asterclient Enhancement
***********************
the parametric studies/distribution should probably be completely redesigned
============================================================================
* create a kind of dict like object which contains a set of similar dicts, but
  not necessery same keys, but similar
* these similar dicts are the parameters for te different studies
* the object provides methods to decide for one study
* it also needs to have a way to retrieve the items of the selected study
* should work something like that:

    In [39]: studies = Studies(
             default={'a':1,'b':'kajhf'},
             studies={
                 'first':{'a':9,'b':'i'},
                 'second':{'a':0},
                 }
             )
    In [40]: studies.add_study(
                 name='third',
                 parameters={'b':'hallllllo'}
             )

    In [41]: studies.select_study('second')
    In [42]: study['a']
    Out[42]: 0
    In [43]: studies['b']
    Out[43]: 'kajhf'

* it should also be possible to select the studies by numbers
* best would be to also have the instantiation in a seperate module which can
  be imported by other modules for different calculations, eg:
  strain-stress-linear, buckling, ...
                 
* the selection of the study should be done over sys.args, since the tudies
  nieed to be available for all calculations independetnly, or use the the
  injection you use now o select the study

Inspect Code Aster Objects
**************************

One can just run a preparation for a calcualtion and drop itself into an ipdb
session by import ipdb;ipdb.set_trace()

Aster
*****
to run aster here, just make sure the REPE_OUT directory is empty and then fire
up::

    ./asteru Python/Execution/E_SUPERV.py -eficas_path ./Python -commandes \
    fort.1  -num_job 16432 -mode interactif \
    -rep_outils /data/opt/aster/outils \
    -rep_mat /data/opt/aster/STA11.2/materiau -rep_dex \
    /data/opt/aster/STA11.2/datg  -memjeveux 64.0 -tpmax 900


Logical unit numbers
********************
you can get a list of them from the file ``unites_logiques.tcl`` in the
``$ASTER_ROOT/lib/astk/`` directory.
