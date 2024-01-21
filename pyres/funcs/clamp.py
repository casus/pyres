import numpy as np

def clamp(val, minval=None, maxval=None):
    """
    Clamp an nd-array between minval and maxval.
    
    Parameters:
        val: Input array.
        minval: Any value of val smaller than minval will be made equal to minval.
        maxval: Any value of val larger than maxval will be made equal to maxval.
    
    Returns:
        val: Clamped value.
    """
    
    if minval is None:  # no lower bound
        val[val > maxval] = maxval
    elif maxval is None:  # no upper bound
        val[val < minval] = minval
    else:
        val[val > maxval] = maxval
        val[val < minval] = minval
        
    return val


