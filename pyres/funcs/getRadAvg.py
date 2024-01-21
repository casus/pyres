import numpy as np
from scipy.ndimage import zoom
from .im2pol import im2pol

def getRadAvg(im):
    if len(im.shape) != 2:
        raise ValueError('getRadAvg supports only 2D matrix as input')
    
    # Check if the input image is square, if not, crop it
    if im.shape[0] != im.shape[1]:
        N = min(im.shape[0], im.shape[1])
        im = im[int(im.shape[0]/2 - N/2): int(im.shape[0]/2 + N/2), 
                int(im.shape[1]/2 - N/2): int(im.shape[1]/2 + N/2)]
    
    # Convert to polar and compute mean
    r = np.mean(im2pol(im), 1)

    # Resizing the result to match MATLAB's ceil behavior
    target_length = int(np.ceil(im.shape[1]/2))
    r = zoom(r, (target_length / len(r),), order=1)  # bilinear interpolation

    return r
