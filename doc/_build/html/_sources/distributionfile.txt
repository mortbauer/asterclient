.. _distributionfile:

Distributionfile
****************
The distributionfile is a simple python file which can contain any valid python
code, but needs at least to provide a variable called ``parameters`` which is a
list holding the various parameters for the parametric studies. The list must
contain tuples with two entries, where the first entry is a string containing
the name of that study and as second entry a python dict containing all
parameters. It could for example look like:

.. code-block:: python

    #coding=utf-8

    parameters = [
        ('study_A',{'a':3,'b':2}),
        ('study_B',{'a':1,'b':19})
        ]

In your commandfile you could acces these variables now through ``params['a']``
or ``params['b']`` respectively, assuming that you have specified the
distributionfile correctly in the :ref:`profile`.

