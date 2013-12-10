.. _profile:

Profile File
############
The profile file is comparable to code aster export file used by asrun_ and
astk. The difference to that format is basically it's format which has to be
valid yaml_, and is hopefully easier in it's data structure. The available
keys are described below in detail.  A full example profile file can be
downloaded :download:`here <../examples/basic/profile.yml>`.

.. confval:: project

    This specifies the projectname for the calculations and has only
    informative character and is optional.

.. confval:: srcdir

    Here you specify the source directory for the calculation, if you omit it
    it will point to the directory currently run the client and is therfore
    optional.

.. confval:: outdir

    Here you specify win which directory the calculation results should be
    saved to. Be careful, asterclient will overwrite files or directories if
    the have have same name as some results. If you specify a relative path it
    will be considered relative to the directory you run asterclient from.

.. confval:: meshfile

    The meshfile key specifies the path to the meshfile for the calculations,
    if relative to the :confval:`srcdir` or absolute.
    
.. confval:: calculations

    This is a list of all known calculations, for example some stress
    calculation or some fatigue calculation of the same project and the same
    mesh. Every calclation needs a name and a commandfile, fr example::

        - name: "stress"
          commandfile: "stress.comm"
        - name: "fatigue"
          commandfile: "fatigue.comm"

    This will provide to calculations named stress and fatigue with the
    associated commandfiles. If you want to run some calculation which needs
    some results of some other calculation as it's input you need to specify
    the poursuite key, for example::

        - name: "post"
          commandfile: "post.comm"
          poursuite: "stress"

    This would tell asterclient that the calculation ``post`` needs the results
    of the calculation ``stress`` as it's input, of course therefore you need
    first to calculate ``stress`` before you can calculate ``post``.

    .. confval:: name

        The name of the calculation. 

    .. confval:: commandfile

        The commandfile associated with the specified calculation.

    .. confval:: resultfiles

        A list of additional (in addition to the standard protocol output and
        glob.1 and pick.1) result files. You specify a file with a name and
        Logical Unit Number LU (see codeasterglossary_ under UNITE), for example::

            - example.med: 80
            - buckling.med: 81

        Which would specify two files one with the name example.med and a LU
        numbe rof 80 and one with the name buckling.med and a LU number of 81.
        They could can now be refered to in the commandfile of the calculation
        for example like::

            IMPR_RESU(FORMAT='MED',
                    UNITE=81,
                    RESU=....
                    )

        If you want to write some result files through python then you also
        need to add these files here other wise they won't get copied from the
        work directory to the result directory, you can also use globbing here.
        For example::

            - protocol: ".rst"

        would result into copying of the file ``protocol.rst`` from the working
        directory to the result directory.::

            - protocol: "*.rst"

        would result in copying all files starting with ``protocol`` and with
        the extension ``rst`` to the result directory, if there isn't any file
        found or if the file is empty you get a warning.


            
.. confval:: distributionfile

    If you want to run a parametric study, which means that you have
    calculations which need basically the same commandfile but with different
    values, the you just specify a distributionfile with tis confval. The
    explanation on how the distributionfile needs to look like see
    :ref:`distributionfile`. For information on how to use the specified parameters
    in the commandfile see :ref:`commandfile`.



.. _asrun: http://manpages.ubuntu.com/manpages/precise/man1/as_run.1.html
.. _codeasterglossary: http://www.code-aster.org/wiki/doku.php?id=en:p01_util:p120_terms
.. _yaml: http://en.wikipedia.org/wiki/YAML

