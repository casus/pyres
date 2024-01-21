import numpy as np

def getCorrcoef(I1, I2, c1=None, c2=None):
    """
    Return the normalized correlation coefficient expressed in Fourier space.

    Parameters:
        I1: Complex Fourier transform of image 1
        I2: Complex Fourier transform of image 2
        c1: Optional normalization constant for I1
        c2: Optional normalization constant for I2

    Returns:
        cc: Normalized cross-correlation coefficient
    """

    if c2 is None:
        c2 = np.sqrt(np.sum(np.abs(I2)**2))
    if c1 is None:
        c1 = np.sqrt(np.sum(np.abs(I1)**2))

    cc = np.sum(np.real(I1 * np.conj(I2))) / (c1 * c2)
    cc = np.floor(1000 * cc) / 1000

    return cc

