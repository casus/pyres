import numpy as np
from .linmap import linmap

def apodImRect(input_img, N):
    """
    Apodize the edges of a 2D image

    Parameters:
        input_img: numpy.ndarray
            Input image
        N: int
            Number of pixels of the apodization

    Returns:
        tuple of numpy.ndarray: Apodized image, Mask used to apodize the image
    """
    if len(input_img.shape) == 3 and input_img.shape[2] == 1:
        input_img = np.squeeze(input_img, axis=2)

    Ny, Nx = input_img.shape

    x = np.abs(np.linspace(-Nx/2, Nx/2, Nx))
    y = np.abs(np.linspace(-Ny/2, Ny/2, Ny))

    mapx = x > Nx/2 - N
    mapy = y > Ny/2 - N

    val = np.mean(input_img)

    d = (-abs(x) - np.mean(-abs(x[mapx]))) * mapx
    d = linmap(d, -np.pi/2, np.pi/2)
    d[~mapx] = np.pi/2
    maskx = (np.sin(d) + 1) / 2

    d = (-abs(y) - np.mean(-abs(y[mapy]))) * mapy
    d = linmap(d, -np.pi/2, np.pi/2)
    d[~mapy] = np.pi/2
    masky = (np.sin(d) + 1) / 2

    # Convert to 2D
    mask = np.outer(masky, maskx)

    output_img = (input_img - val) * mask + val

    return output_img, mask
