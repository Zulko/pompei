Pompei
========

Pompei is a Python package to make mosaics from movie frames. In the following mosaics, a particular frame of a movie is reconstructed using other frames from the same movie. Click on the icon in the upper right corner for full resolution.

.. raw:: html

         <iframe class="imgur-album" width="100%" height="600px" frameborder="0" src="http://imgur.com/a/Bycb3/embed?background=fefefa&text=111111&link=4e76c9"></iframe>


Pompei is an open source software written originally by Zulko_ and released under the MIT licence. The project is hosted in Github_ where you can propose commits or ask for help.


.. raw:: html

    <a href="https://twitter.com/share" class="twitter-share-button"
    data-text="Create mosaics from movie frames with Python and Pompei" data-size="large" data-hashtags="PythonPompei">Tweet
    </a>
    <script>!function(d,s,id){var js,fjs=d.getElementsByTagName(s)[0],p=/^http:/.test(d.location)?'http':'https';
    if(!d.getElementById(id)){js=d.createElement(s);js.id=id;js.src=p+'://platform.twitter.com/widgets.js';
    fjs.parentNode.insertBefore(js,fjs);}}(document, 'script', 'twitter-wjs');
    </script>
    
    <iframe src="http://ghbtns.com/github-btn.html?user=Zulko&repo=pompei&type=watch&count=true&size=large"
    allowtransparency="true" frameborder="0" scrolling="0" width="152px" height="30px" ></iframe>

Installation
--------------

Then install Pompei, either by unzipping the source code in a directory and typing in a terminal: ::

    sudo python setup.py install

Or, if you have PIP installed: ::

    (sudo) pip install ez_setup pompei


Pompei depends on MoviePy_ which will be automatically installed during the installation of Pompei.

How it works
-------------

The mosaic algorithm proceeds as follows:

- Many frames of the movie (for instance one every 5 second) are extracted and downsized to make small thumbails. This takes a few minutes, but you will only need to do it once for each movie. The remaining steps takes less than 30 seconds in total.
- The image from which the mosaic will be made is divided into small rectangular regions.
- At first, each region is matched with the thumbnail that fits it best. To speed up computations Pompei reduces each subregion of the image, and each thumbnail extracted in step 1, to a few average pixels (for instance, 3x4 pixels), called the "signature" of the image. An image region and a thumbnail are said to "fit" well when there signatures are similar.
- The thumbnails associated with some regions are then changed to ensure that no thumbnail is represented too many times (first pass), and that thumbnails whitch are near in time in the original movie are well spread apart in the final picture (second pass).
- Finally all the thumbnails are assembled on a big canvas to form the mosaic, which is written to a file.

Some pictures can be well *mosaiced* using the frames of a given movie, but some can't. To help you choose the "right" image for your mosaic, Pompei provides a method (depending on the package Scikit-Learn) to find which thumbnail image extracted from the movie during the first step can be well reconstructed with the other thumbnails. This is not fully tested but seems to work quite well, here is the algorithm:

- Pool together the colors of the different images, and find the K=10 colors which represent best the general colors of the movie, using a K-means clustering algorithm.
- For each thumbnail, replace each of its pixels by the nearest of the K selected colors. If the color change didn't modify much the thumbnail, it means that that the color in this frame are well represented in the movie in general, and a mosaic could be made from this image. Therefore the difference between the original thumbnail and its color-quanitzed version gives you a *score*. Frames with the lowest score are the best candidates for a mosaic.


Example of code
------------------

In this script we extract frames from a movie and detect 50 frames which could make a good mosaic, then we select one particular frame and turn in into a mosaic. ::

    from pompei import MovieFrames
    
    # The next command is only used once to extract frames from the movie.
    
    movieframes = MovieFrames.from_movie('gladiator.flv', # a video file
        foldername='gladiator', # where frames are extracted
        fps=1.0/5, # take one frame every 5 seconds.
        crop=(10*60,10*60), # cut out first and last 10 min.
        sig_dim=(3,3)) # frames signatures: 3x3 pixels
    
    # For when the movie frames are already extracted:
    
    movieframes = MovieFrames(foldername='gladiator')
    
    
    # (optional) find the thumbnails that would make great mosaics.
    # They are extracted and saved with the corresponding time and score
    # in the title, so that the original image can easily be retrieved
    
    for i, score in movieframes.find_mosaicable_frames()[:50]:
        time = movieframes.times[i]
        movieframes.extract(i, "gladiator_%d_%.2f.png"%(time,score))
    
    
    
    # Load the image that you are going to turn into a mosaic, either
    # from an image file or a frame from a video file
    
    from moviepy.editor import VideoFileClip, ImageClip
    image = ImageClip("some_image.jpeg") # or next line:
    image = VideoFileClip("gladiator.flv").get_frame('01:26:43')
    
    
    # Make a mosaic from the frame. Many options available.
    
    movieframes.make_mosaic( image,
      "gladiator_mosaic.png", # output file (will be large and heavy !)
      frames_in_width=60, # 'image' is reconstructed with 60x60 thumbails
      max_occurences=10, # wished maximal occurence of a thumbnail.
      maxiter=3000,     # abandomn optimization after 3000 steps
      spatial_window=2, # frames at a distance of 2 or less must be
      time_window=3)    # separated by 3 indices (here 15 movie seconds)

Command line API
-----------------

**Not yet implemented, these are just thoughts.** When Pompei is installed it also provides the following command line interface: ::

    >>> pompei.py -h #to get detailed help
    >>> pompei.py video_to_folder my_video.mp4 my_folder --signatures=3x3
    >>> pompei.py find_best_frames my_folder --lum_min=50 --nframes=50
    >>> pompei.py extract_image my_video.mp4 '01:02:30.10' myframe.png
    >>> pompei.py make_mosaic folder myframe --res=50 --max_occurences=12

Customizing Pompei
-------------------

There are many parameters you can tweak in Pompei (see API below), and the code is very modular, which makes it easy to change any part of the matching algorithm. The simplest way to do this it to first go see in the module's code which method performs the task that you wish to customize, then, in your script, overwrite this method, for instance by creating a subclass of MovieFrames: ::
    
    class MovieFrames2(MovieFrames):
        def _find_best_matches(self, a, b):
            bla = 1
            # write your own version of this function


Reference manual
-----------------


.. autoclass:: pompei.MovieFrames
   :members:

.. _Zulko: https://github.com/Zulko/
.. _MoviePy: http://zulko.github.io/moviepy
.. _Github: https://github.com/Zulko/pompei

