import os
import numpy as np
from moviepy.editor import ImageClip, VideoFileClip
from tqdm import tqdm

def split_array(arr, nh, nw):
    """ Splits a 2D-ish array into a mosaic of smaller arrays.

    In short, this array:

    [ [ A ] ]

    is splitted like this

    [ [  [A11] [A12] ]
      [  [A21] [A22] ] ]

    """
    h,w,_ = arr.shape
    dh, dw = h/nh, w/nw 
    return np.array([[arr[dh*i:dh*(i+1),dw*j:dw*(j+1)]
                      for j in range(nw)]
                      for i in range(nh)])


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


def compute_signatures_in_image(image, nh_signatures, nw_signatures,
                              nh_image=None, nw_image=None,
                              dh_image=None, dw_image=None ):
    if dh_image is not None:
        w, h, _ = image.shape
        nh_image, nw_image = h/dh_image, w/dw_image

    signature = lambda im: colors_signature(im, nh_signatures, nw_signatures) 
    return map_array(signature, split_array(image, nh_image, nw_image))

def save_signatures(signatures, folder):
    target = os.path.join(folder, 'signatures.txt')
    np.savetxt(target, np.array(signatures))



def compute_signatures_in_folder(folder, nh, nw):
    signatures = []
    for filename in os.listdir(folder):
        filename = os.path.join(folder, filename)
        img = ImageClip(filename).get_frame(0)
        signatures.append(colors_signature(img, nh, nw))
    save_signatures(signatures, folder)
    return signatures

def movie_to_folder(filename, foldername, tt=None, fps=None, resize_factor=0.2,
                    signatures_nh=2, signatures_nw=2, subclip=None):
    
    if not os.path.exists(foldername):
        os.mkdir(foldername)

    clip = VideoFileClip(filename, audio=False).resize(resize_factor)
    if subclip is not None:
        clip = clip.subclip(*subclip)

    if tt is None:
        tt = np.arange(0,clip.duration, 1.0/fps)
    
    signatures = []
    for i,t in tqdm(enumerate(sorted(tt)), total=len(tt)):
        target = os.path.join(foldername, "%06d.png"%i)
        clip.save_frame(target, t)
        signatures.append(colors_signature(clip.get_frame(t), signatures_nh,
                                           signatures_nw))
    save_signatures(signatures, foldername)
    return signatures

def best_match_index(frame_signature, other_frames_signatures, forbidden=None):
    diffs = np.mean((1.0*frame_signature-other_frames_signatures)**2, axis=1)
    if forbidden is not None:
        for ind in forbidden:
            diffs[ind] = 4*256**2 # this eliminates de facto the indice
    ind = np.argmin(diffs)
    return ind

def get_image_signatures_from_folder(folder, nh=None, nw=None):
    """ Returns the signatures from folder/signatures.txt if it exists, else
        computes the signature of each image (may take long), creates the
        folder/signatures.txt file, and returns the signatures.

        The signatures are a Numpy array:
        [ [signature_image1],
          [signature_image2],
          [signature_image3]
          ...]
    """
    target = os.path.join(folder, 'signatures.txt')
    if os.path.exists(target):
        return np.loadtxt(target)
    else:
        return compute_signatures_in_folder(folder, nh, nw)



def find_best_matches(image_signatures, other_frames_signatures,
              npasses=500, goal=20, percentile_changed = 30, time_window=3,
              spatial_window=5):
    """




    """

    from collections import Counter
    
    # first guess, will be refined just after
    h, w, _ = image_signatures.shape
    image_signatures = np.vstack(image_signatures)
    print image_signatures.shape
    best_match = lambda sig: best_match_index(sig, other_frames_signatures)

    # FIRST PASS: WE CHOOSE THE BEST FRAME FOR EVERY IMAGE REGION

    best_matches = np.array([best_match_index(sig, other_frames_signatures)
                    for sig in image_signatures])
    
    # SECOND PASS: WE ITERATIVELY REPLACE FRAMES SO THAT SOME FRAMES ARE NOT
    # OVERUSED
    
    forbidden = []
    npass, number = 0, 1000
    print "Now starting occurences minimization"
    while (npass < npasses) and (number > goal):
        ind, number = Counter(best_matches).most_common(1)[0]
        forbidden.append(ind)
        indices = (best_matches==ind).nonzero()[0]
        indices_to_keep = indices[::3]
        indices_to_change = [i for i in indices if not i in indices_to_keep]
        for ind in indices_to_change:
            best_matches[ind] = best_match_index( image_signatures[ind],
                                                  other_frames_signatures,
                                                  forbidden)
        npass += 1
    print "Done. The max occurence is now %d."%number

    best_matches = best_matches.reshape((h,w))


    # THIRD PASS: WE AVOID THAT SIMILAR FRAMES HAPPEN IN THE SAME REGION
    
    if spatial_window != 0:
        max_inds = len(other_frames_signatures)
        print "Now starting last pass (eliminate similar neighbours)"
        counter=0
        for i in tqdm(range(h)):
            for j in range(w):
                ind = best_matches[i,j]
                x1, x2 = max(0, j-spatial_window), min(w, j+spatial_window)
                y1, y2 = max(0, i-spatial_window), min(h, i+spatial_window)
                spatial_indices = list(best_matches[y1:y2, x1:x2].flatten())
                #print x1, x2, y1, y2
                spatial_indices.pop(spatial_indices.index(ind))
                forbidden_indices = sum([range(max(0,k-time_window),
                                         min(max_inds-1, k+time_window+1))
                                        for k in spatial_indices], [])
                ff = [k for k in forbidden if k!=ind]
                forbidden_indices = ff+list(set(forbidden_indices))
                new_ind = best_match_index( image_signatures[w*i+j],
                                                  other_frames_signatures,
                                                  forbidden_indices)
                if new_ind != ind:
                    counter +=1

                best_matches[i,j] = new_ind
        
        print "Last pass changed %d frames"%counter
        ind, number = Counter(best_matches.flatten()).most_common(1)[0]
        print "Now the most represented frame is represented %d times."%number


    return best_matches


def best_matches_to_image(filename, best_matches, folder):

    def get_image(index):
        target = os.path.join(folder, "%06d.png"%index)
        return ImageClip(target).get_frame(0)

    unique_inds = sorted(list(set(best_matches.flatten())))
    D = {ind: get_image(ind) for ind in tqdm(unique_inds)}


    image = ImageClip(stack_array(map_array(D.get, best_matches)))
    image.save_frame(filename)