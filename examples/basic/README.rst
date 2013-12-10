Example Bikeframe
#################
To run the bikeframe example type::

    asterclient run -p profile.yml

To get information on the project run::

    asterclient info -p profile.yml

This will run all calculations on all studies. If you just wanna run the
``main`` calculation, type::

    asterclient run -p profile.yml -c main

Or if you just want to run the ``main`` calculation on the
``lenkkopfsteifigkeit`` study type::

    asterclient run -p profile.yml -c main -s lenkkopfsteifigkeit
    
