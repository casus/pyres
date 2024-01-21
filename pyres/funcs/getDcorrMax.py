import numpy as np

def getDcorrMax(d):
    """
    Return the local maxima of the decorrelation function d

    Parameters:
        d: Decorrelation function (1D numpy array)

    Returns:
        ind: Position of the local maxima
        A: Amplitude of the local maxima
    """

    A = np.max(d)
    ind = np.argmax(d)
    t = d.copy()

    # Arbitrary peak significance parameter imposed by numerical noise
    # This becomes relevant especially when working with post-processed data
    dt = 0.001

    while ind == len(t) - 1:  # Python uses 0-based indexing
        t = t[:-1]
        
        if len(t) == 0:
            A = 0
            ind = 0
        else:
            A = np.max(t)
            ind = np.argmax(t)
            
            # Check if the peak is significantly larger than the former minimum
            if t[ind] - np.min(d[ind:]) > dt:
                break
            else:
                t[ind] = np.min(d[ind:])
                ind = len(t) - 1

    return ind, A


