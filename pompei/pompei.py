import os
import sys
import numpy as np
from collections import Counter
from tqdm import tqdm
from moviepy.video.io.ffmpeg_reader import ffmpeg_read_image
from moviepy.video.io.ffmpeg_writer import ffmpeg_write_image
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.fx.resize import resize

# PARAMETERS FOR VERBOSITY

PARAMETERS = {"VERBOSE":True}

"""
These lines enable to switch the libraries' parameters, for example:
>>> from pompei import set_parameter
>>> set_parameter("VERBOSE", False)
"""

def set_parameter(parameter, value):
    PARAMETERS[parameter] = value

def verbose_print(s):
    if PARAMETERS['VERBOSE']:
            sys.stdout.write(s)
            sys.stdout.flush()

def vtqdm(l, **kw):
    """ applies tqdm (progress bar) to the list to iterate, only if
        VERBOSE=True. """
    return tqdm(l, **kw) if PARAMETERS['VERBOSE'] else l

# ARRAY OPERATIONS

def split_array(arr, nh, nw):
    """ Splits a 2D-ish array into a mosaic of smaller arrays.

    In short, this array:

    [ [ A ] ]

    is splitted like this

    [ [  [A11] [A12] ]
      [  [A21] [A22] ] ]

    """
    h,w,_ = arr.shape
    dh, dw = 1.0*h/nh, 1.0*w/nw 
    return [[ arr[ int(np.round(dh*i)): int(np.round(dh*(i+1))),
                     int(np.round(dw*j)): int(np.round(dw*(j+1)))]
              for j in range(int(nw))] for i in range(int(nh))]


def stack_array(arr):
    """ Reconstitutes a splitted array. Kind of the inverse of split-array.

    In short, this array
    
    [ [  [A11] [A12] ]
      [  [A21] [A22] ] ]

    becomes this:

    [ [ A11 A12 ]
      [ A21 A22 ] ]
    """
    return np.vstack([np.hstack(line) for line in arr])


def map_array(fun, arr):
    """ applies a function to each element of a 2D-ish array.

    So this array:

    [ [  [A11] [A12] ]
      [  [A21] [A22] ] ]

    becomes this:

    [ [  [fun( A11 )] [fun( A12 )] ]
      [  [fun( A21 )] [fun( A22 )] ] ]
    """

    return np.array([[fun(e) for e in line] for line in arr])


def mean_color(arr):
    """ Returns [R,G,B], mean color of the WxHx3 numpy image ``arr``. """
    return arr.mean(axis=0).mean(axis=0)


def colors_signature(image, nh, nw):
    """ Returns the signature of the image.

    [ [ RGB1, RGB2, RGB3]
      [ RGB4, RGB5, RGB6] ]

    and returns [R1 G1 B1 R2 G2 B2 R3 G3 B3... R6 G6 B6]
    """
    return map_array(mean_color, split_array(image, nh, nw)).flatten()




class MovieFrames:
    """
    Base class in Pompei. Objects represent a series of thumbnail, which are
    produced from a movie with MovieFrames.from_movie(), are analyzed with
    MovieFrame.find_mosaicable_frames(), and are used to reconstitute mosaics
    with MovieFrames.make_mosaic().
    """

    def __init__(self, folder):
        filename = lambda s: os.path.join(folder,s) 
        self.folder = folder
        self.signatures = np.loadtxt(filename("signatures.txt"))
        self.signatures_dim = np.loadtxt(filename("signatures_dim.txt"))
        self.times = np.loadtxt(filename("times.txt"))
        self.imagestack = ffmpeg_read_image(filename("all_frames.png"),
                                            with_mask=False)
        self.im_h, self.dw, _ = self.imagestack.shape
        self.sig_h, self.sig_w = self.signatures.shape
        self.dh = self.im_h/self.sig_h

    def __getitem__(self, index):

        return self.imagestack[index*self.dh:(index+1)*self.dh]

    def __len__(self):
        return self.sig_h
        
    def __iter__(self):

        for i in range(self.sig_h):
            yield self[i]
            
    def extract(self, index, filename=None):
        if hasattr(index, '__iter__'):
            for ind in index:
                extract(self, ind, filename)

        if filename is None:
            filename = os.path.join([self.folder, "extracted", "%05d.png"%index])

        ffmpeg_write_image(filename, self[index])

    @staticmethod
    def from_movie(filename, foldername=None, tt=None, fps=None,
                   crop=(0,0), thumbnails_width= 120, sig_dim=(2,2)):
        """
        Extracts frames from a movie and turns them into thumbnails.

        Parameters
        ==========

        filename
          Name of the legally obtained video file (batman.avi, superman.mp4,
          etc.)

        foldername
          The extracted frames and more infos will be stored in that directory.

        tt
          An array of times [t1, t2...] where to extract the frames. Optional if

        crop
          Number of seconds to crop at the beginning and the end of the video,
          to avoid opening and en credits.
          (seconds_cropped_at_the beginning, seconds_cropped_at_the_end)

        thumbnails_width
          Width in pixels of the thumbnails obtained by resizing the frames of
          the movie.

        sig_dim
          Number of pixels to consider when reducing the frames and thumbnails
          to simple (representative )signatures.
          sid_dim=(3,2) means 3x2 (WxH) pixels.



        """
        if foldername is None:
            name, ext = os.path.splitext(filename)
            foldername = name

        if not os.path.exists(foldername):
            os.mkdir(foldername)

        clip = VideoFileClip(filename).fx( resize, width=thumbnails_width)
        if not os.path.exists(foldername):
            os.mkdir(foldername)

        if tt is None:
            tt = np.arange(0,clip.duration, 1.0/fps)
            t1, t2 = crop[0], clip.duration-crop[1]
            tt = tt[ (tt>=t1)*(tt<=t2)]

        signatures = []

        result = np.zeros((clip.h*len(tt), clip.w, 3))

        for i,t in vtqdm(enumerate(sorted(tt)), total=len(tt)):
            frame= clip.get_frame(t)
            result[i*clip.h:(i+1)*clip.h] = frame
            signatures.append(colors_signature(frame, sig_dim[0], sig_dim[1]))

        target = os.path.join(foldername, "all_frames.png")
        ffmpeg_write_image(target, result)

        for (obj, name) in [(signatures, 'signatures'), (tt, 'times'),
                            (sig_dim, 'signatures_dim')]:
            target = os.path.join(foldername, name+'.txt')
            np.savetxt(target, np.array(obj))

        return MovieFrames(foldername)
    
    def find_mosaicable_frames(self, nclusters = 8, luminosity_threshold=50):
        """ Finds the frames whose colors resemble most the main colors of the
            movie, and thus would make good candidates for a mosaic. This is
            highly experimental.

            Requires Scikit-learn installed.

            Returns  [(frame1,score1), (frame2, score2)...] where the lowest
            scores indicates frames better suited for a mosaic.
        """

        try:
            from sklearn.cluster import KMeans
        except ImportError:
            raise ImportError("sklearn.Cluster not found. You must install"
                  " Scikit-learn to be able to use 'find_mosaicable_frames'.")
        
        movie_colors = np.vstack(self.signatures.reshape((self.sig_h,self.sig_w/3,3)))

        movie_kmeans = (KMeans(n_clusters=nclusters, random_state=0).fit(movie_colors))
        
        def score(im):
            """ Returns the sum of the squared distance of each pixel to the
            nearest color in the movie_kmeans set."""
            imcolors = np.vstack(im).astype(float)
            return sum(movie_kmeans.transform(imcolors).min(axis=1)**2)
        
        scores = [score(im) for im in vtqdm(self)]
        sorted_scores= sorted([(i,score) for (i,score) in enumerate(scores)
                               if self[i].mean()>luminosity_threshold],
                               key=lambda s:s[1])
        return sorted_scores


    def _score(self, best_matches, image_signatures):
        """ Returns the squared distance (matching error) between a frame and
            a signature. """ 

        f = lambda ind: self.signatures[ind]
        matches_signatures =np.vstack( map_array(f, best_matches) )
        return np.sqrt(np.sum((matches_signatures - image_signatures)**2))


    def _best_match_index(self, frame_signature, forbidden=None):
        """ Finds the movie frame that matches best the given signature. Will
            try all movie frames except the ones designated as forbidden (which
            are frames already too frequent in the mosaic). """
        
        # an array of all the matching distances frame-signature
        diffs = np.mean((1.0*frame_signature-self.signatures)**2, axis=1)

        if forbidden is not None:
            for ind in forbidden:
                # Next line eliminates "de facto" the index from the possible
                # candidates for a best match.
                diffs[ind] = len(frame_signature)*500000

        ind = np.argmin(diffs)
        return ind

    def _minimize_occurences(self, best_matches, image_signatures, maxiter,
                             max_occurences='auto', matching_quality=.7):
        """ Diversifies the mosaic by replacing the overused frames by different
        (and non optimal) frames.

        Parameters
        ==========

        best_matches
           The current 'solution': a 2D array of frames numbers corresponding to
           the optimal currently chosen to represent the corresponding region
           in the mosaic.

        image_signatures
           The 2D array of signatures of the image to turn into a mosaic.

        maxiter
           Number of loops after which to stop (to avoid the program running
           indefinitely).
        
        max_occurences
            The algorithm will stop when the most frequent frame is represented
            max_occurences time. If you leave it to auto, the algorithm stops
            after `maxiter` operations.
        
        matching_quality
            The program will stop when the current total matching error is less
            than initial_score / matching_quality. this is to avoid that the
            program degrades the quality of the mosaic too much. 

        """

        shape = best_matches.shape
        initial_score = self._score(best_matches, image_signatures)
        if max_occurences == 'auto':
            max_occurences = -1

        verbose_print("Minimizing occurences...")
        best_matches = best_matches.flatten()
        forbidden = []
        npass, number, ratio = 0, 1000, 2
        while ((npass < maxiter) and (number > max_occurences) and
               (ratio > matching_quality)):
            ind, number = Counter(best_matches).most_common(1)[0]
            forbidden.append(ind)
            def score(i):
                _score = lambda index : np.sum( (image_signatures[i] -
                                                 self.signatures[index])**2) 
                score_before = _score(best_matches[i])
                new_ind = self._best_match_index( image_signatures[i], forbidden)
                score_after = _score(new_ind)
                return np.sqrt(score_before) - np.sqrt(score_after)

            indices = sorted ( (best_matches==ind).nonzero()[0], key=score)
            ix = max_occurences if (max_occurences!=-1) else len(indices)/2
            indices_to_change = indices[ix:]
            for ind in indices_to_change:
                best_matches[ind] = self._best_match_index(
                                          image_signatures[ind], forbidden)    
            if (max_occurences == -1):
                new_score = self.score(best_matches.reshape(shape),
                                             image_signatures)
                ratio = 1.0*initial_score/new_score
            npass += 1

        new_score = self._score(best_matches.reshape(shape),
                                             image_signatures)
        ratio = 1.0*initial_score/new_score        

        verbose_print("Done. Max occurence:%d, quality:%.04f.\n"%(number, ratio))

        return best_matches.reshape(shape), number



    def _eliminate_similar_neighbors(self, best_matches, image_signatures,
                             max_occurences, spatial_window, time_window):
        """ Replaces some frames to avoid that frames too close in the movie end
        up too close in the mosaic.
        """

        counter  = Counter(best_matches.flatten())
        h,w = best_matches.shape
        max_inds = len(self)
        verbose_print("Eliminating similar neighbours...")
        ctr=0
        for i in vtqdm(range(h)):
            for j in range(w):
                ind = best_matches[i,j]
                x1, x2 = max(0, j-spatial_window), min(w, j+spatial_window)
                y1, y2 = max(0, i-spatial_window), min(h, i+spatial_window)
                spatial_indices = list(best_matches[y1:y2, x1:x2].flatten())
                spatial_indices.pop(spatial_indices.index(ind))
                forbidden_indices = sum([range(max(0,k-time_window),
                                         min(max_inds, k+time_window+1))
                                        for k in spatial_indices], [])
                
                # frames that are forbidden for the given position, i.e. frames
                # that are too near in time from the neighbouring frames.
                ff = [k for (k,v) in counter.items()
                      if (v>=max_occurences) and (k!=ind)]
                forbidden_indices = ff+list(set(forbidden_indices))
                new_ind = self._best_match_index( image_signatures[w*i+j],
                                                  forbidden_indices)
                
                if new_ind != ind:
                    ctr +=1
                    counter[new_ind] += 1

                best_matches[i,j] = new_ind
        verbose_print("Done. %d frames changed\n"%ctr)

        return best_matches



    def _find_best_matches(self, image, frames_in_width=50, max_occurences='auto',
                           maxiter=500, time_window=2, spatial_window=2,
                           matching_quality=0.7):
        """

        See method .make_mosaic


        """
        w = frames_in_width

        im_h, im_w, _ = image.shape

        h = int(np.round(w* (1.0*im_h / im_w) * (1.0*self.dw/ self.dh)) )

        signature = lambda im: colors_signature(im, *self.signatures_dim)
        image_signatures = map_array(signature, split_array(image, h, w))

        # first guess, will be refined just after
        image_signatures = np.vstack(image_signatures)

        # FIRST PASS: WE CHOOSE THE BEST FRAME FOR EVERY IMAGE REGION

        best_matches = np.array([self._best_match_index(sig)
                                 for sig in image_signatures]).reshape((h,w))
        
        # SECOND PASS: MINIMIZE OCCURENCES
        
        best_matches, max_occurences = self._minimize_occurences(best_matches,
                                      image_signatures, maxiter, max_occurences,
                                      matching_quality)
        
        # THIRD PASS: ELIMINATE SIMILAR NEIGHTBORS

        if spatial_window:
            best_matches = self._eliminate_similar_neighbors(best_matches,
                               image_signatures, max_occurences, spatial_window,
                               time_window)

        return best_matches, self._score(best_matches, image_signatures)


    
    def make_mosaic(self, image, outputfile,  frames_in_width=50,
                    max_occurences='auto',
                    maxiter=500, time_window=2, spatial_window=2,
                    matching_quality=0.7):
        """
        Makes a mosaic file from the best_matches found.

        Finds the right frames to reconstitute the given image, in three steps:
        
        1. Find the optimal frame for each subregion of the given image.
        2. Change the frames so as to avoid the same frames being used too many
          times.
        3. Change the frames so that frames that are near in time in the original
          movie will not appear near from each other in the final mosaic.

        Parameters
        -----------

        image
          The (RGB WxHx3 numpy array) image to reconstitute.

        frames_in_width
          How many frames are in the width of the picture (determines the
          'resolution' of the final mosaic)

        max_occurences
          How many occurences of the same frame are allowed in step 2. If auto,
          this is not taken into account and the algorithm will try and reduce
          the number of occurences until it reaches maxiter iterations or bad
          overall matching quality (see below)

        maxiter
          Number of iterations in step 2 when reducing the number of occurence
          of the most frequent frames.

        matching_quality
            The program will stop when the current total matching error is less
            than (initial_score / matching_quality). this is to avoid that the
            program degrades the quality of the mosaic too much.

        time_window

          The frames will be changed in step 3 so 

        """

        best_matches, score = self._find_best_matches(image, frames_in_width,
                                 max_occurences, maxiter, time_window,
                                 spatial_window, matching_quality)

        nframes = len(list(set(best_matches.flatten())))
        verbose_print("Assembling final picture (%d different frames)."%nframes)
        h, w = best_matches.shape
        dh,dw = self.dh, self.dw

        final_picture = np.zeros((h*dh, w*dw, 3))

        for i in range(h):
            for j in range(w):
                ind = best_matches[i,j]
                final_picture[dh*i:dh*(i+1), dw*j:dw*(j+1)] = self[ind]

        ffmpeg_write_image(outputfile, final_picture.astype('uint8'))
        verbose_print("Finished.\n")