Pompei
=================

Pompei (documentation and examples here_) is a python package and command line utility to generate photo mosaics from movie frames,
like these ones:


.. raw:: html

         <iframe class="imgur-album" width="100%" height="600px" frameborder="0" src="http://imgur.com/a/Bycb3/embed?background=fefefa&text=111111&link=4e76c9"></iframe>


It is an open-source software originally written by Zulko_ and released under the MIT licence. Everyone is welcome to contribute or ask for support on the Github_ project page.

Installation
--------------

Pompei can be installed by unzipping the source code in one directory and using this command: ::

    (sudo) python setup.py install

You can also install it directly from the Python Package Index with this command: ::

    (sudo) pip install ez_setup # <- if you don't have it yet
    (sudo) pip install pompei

Pompei depends on the Python module MoviePy, which will be installed automatically during Pompei's installation.

How it works
-------------

Pompei extracts frames from a movie (say, one every 5 seconds), then finds which frames match best the different parts of the image to reconstruct. For speed, images comparisons are done by reducing each image to a dozen average pixels (called signature of the image). Then an iterative algorithm ensures that each frame does not occur too many times, so that many different frames are used in the end, and a last processing ensures that frames which are near in time are not too close in the mosaic.

This method is empirical, and you are encouraged to modify it to fit your needs (by redefining function ``find_best_matches``).

Use
-----

Upon installation

    pompei -h     # <- displays a detailed help
    pompei batch_extract movie fps folder resize_factor
    pompei extract movie frame_time
    pompei mosaic picture rxry folder outfile goal npasses

See for instance examples/gladiator.sh for the script that generated the image above.

You can use also use Pompei as a Python package: see ``example/gladiator.py`` for the Python version of the same example.

.. _Zulko: https://github.com/Zulko/
.. _here: http://zulko.github.io/pompei
.. _MoviePy: http://zulko.github.io/moviepy
.. _Github: https://github.com/Zulko/pompei