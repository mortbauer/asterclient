
Welcome to Asterclient's documentation!
#######################################

Asterclient is in it's usage very similar to asrun_ program. For a normal run
you will need:

* a :ref:`profile`.yml file which holds the basic data about the calculations (it is
  not needed for a very simple run where specify all relevant data on the
  commandfile, this isn't supported up to now)

* several code aster :ref:`commandfile`'s

* maybe some :ref:`distributionfile` if you want to run a parametric study

Motivation
**********
The motivation for writing asterclient came out of some frustration since the
asrun_ is documented quiet poorly specially on the topic of parametric
studies. I thought it should be much easier and straight forward to run a
paramtric study and in general a simple calculation.

Usage
*****
After succesful :ref:`installation` you need at least the following to run a
calculation [#prereqaster]_:

* profile.yml
* commandfile

For more details read the documentation of the :ref:`profile` file and and the
:ref:`distributionfile` file.

Asterclient currently has two commands available, ``info`` and ``run``, where
info can give you some information on your profile and run does the actual
work, for more info just type ``asterclient info -h`` or ``asterclient run
-h``. The run command needs at least a profile specified, assumed you navigate
into the examples :file:`examples/basic/` directory you just need to type::

    asterclient run -p profile.yml


Basic Example
=============
For a full basic example see the examples directory `example <../../../examples>`_


Detailed Documentation
**********************

.. toctree::
   :maxdepth: 2

   profile
   distributionfile
   commandfile

.. _installation:

Installation
************
The installation is very easy, just download the :ref:`source` and type::

    python setup.py install

.. _source:

Source
******
The sourcecode_ lives at github, feel free to fork mee as much as you like, feedback
is appreciated.

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`


.. _asrun: http://manpages.ubuntu.com/manpages/precise/man1/as_run.1.html
.. _codeasterglossary: http://www.code-aster.org/wiki/doku.php?id=en:p01_util:p120_terms
.. _yaml: http://en.wikipedia.org/wiki/YAML
.. _code aster: http://www.code-aster.org/V2/spip.php?rubrique2
.. _sourcecode: https://github.com/mortbauer/asterclient
.. [#prereqaster] it is further assumed that you have a working `code aster`_ installation on your box
