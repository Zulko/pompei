"""
This is a typical script to reconstruct one frame of a movie using a mosaic
of other frames with the Python package Pompei. It generates this picture of
general Maximus in Gladiator using 1100+ frames of the movie.

http://i.imgur.com/Eoglcof.jpg

This script goes in five steps:

1. Extract one frame every 5 second of the movie. Compute their 'signatures'
2. Extract one special frame (the one to be reconstructed) from the movie.
3. Split this frame into subregions and compute the signature of each region.
4. Run an algorithm to find (using the signatures) wich frames of the movie
   match best with the different regions of the picture to reconstruct.
   The algorithm also ensures that many different frames are used.
5. Assemble the selected best-matching frames into one big picture and save.

The code is well commented to paliate for the lack of documentation. For more,
see the functions doctrings.

"""

from pompei import (movie_to_folder,
                    get_image_signatures_from_folder,
                    compute_signatures_in_image,
                    find_best_matches,
                    best_matches_to_image)

# When comparing the frames of the movie to the regions of the picture to
# reconstruct, each frame and each region will be reduced to Nh x Nw
# zones from which the mean colors are computed. Here we choose 3 x 3.
# The resulting set of 9 colors is called the signature of the region/frame.

signatures_nh=3
signatures_nw=3


### STEP 1 - EXTRACTING THE FRAMES OF THE MOVIE


# For this example we treat gladiator. The result is this mosaic
# http://i.imgur.com/Eoglcof.jpg

foldername = "gladiator" # name of the folder for the frame pictures
filename = 'gladiator.flv' # the video file, from a legally-baught DVD


# The next call extracts the frames from the movie. At the same time it computes
# the signatures of the frames and store them in file gladiator/signatures.txt

# It's pretty long (5 minutes) and should only be done once, then you can
# comment it out if you want to fine-tune the parameters in the next lines.


image_folder_signatures = movie_to_folder(filename, foldername,
                    fps=1.0/5, # take one frame every 5 seconds
                    resize_factor=0.2, # downsize all frames of a factor 1/5
                    signatures_nh=signatures_nh,
                    signatures_nw=signatures_nw,
                    subclip=(5*60,-10*60)) # cut 5-10 minutes to avoid credits.


# Get the signatures of each frame, already computed at the previous step.

image_folder_signatures = get_image_signatures_from_folder(foldername)



### STEP 2 - READING THE IMAGE TO BE RECONSTRUCTED


# Now we load the image to reconstruct. This could be any image but out of
# simplicity we choose one frame frame of the movie, so that it will have the
# same dimensions as the frames that will compose it.
# We take the scene just before "My name is Maximus...".


import moviepy.editor as mpy
image = mpy.VideoFileClip(filename).get_frame('01:26:43.00') # a numpy array.


### STEP 3 - SPLIT THE IMAGE AND COMPUTE THE SIGNATURES OF THE REGIONS


nh = nw = 60
image_signatures = compute_signatures_in_image(image, signatures_nh,
                                               signatures_nw, nh, nw) 


### STEP 4 - FIND THE BEST-MATCHING FRAMES. OPTIMIZE.

# This step is quite quick because we work with signatures (i.e. reduced
# version of the images.

# The algorithm first attributes to each region of the final picture the movie
# frame that matches best. Some frames will be used more than once.
# Then, goal=5 means that the algorithm will iteratively diversify the frames 
# used until the most used frames is used 5 times or less.
# npasses=3000 tells the algorithm to give up after 3000 iterations if it
# cannot reach its goal of 5. Choosing a lower npasses (like npasses=100) can be
# good sometimes to avoid over-diversification.

best_matches = find_best_matches(image_signatures, image_folder_signatures,
                                 npasses=3000,goal=5)


### STEP 5 - ASSEMBLE THE FRAMES INTO ONE BIG PNG FILE

# This produces the final picture: gladiator.png
# This will take long and produce a heavy PNG (50Mo) which can then be
# downsized by converting it to JPEG.

best_matches_to_image("%s.png"%foldername, best_matches, foldername)