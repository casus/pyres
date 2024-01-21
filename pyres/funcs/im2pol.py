import numpy as np
from scipy.interpolate import interp2d

def im2pol(imC, max_resolution=1024):
    if imC.shape[0] > max_resolution or imC.shape[1] > max_resolution:
        raise ValueError("Input image is too large. Consider downsampling before processing.")

    rMin = 0
    rMax = 1

    Ny, Nx = imC.shape
    xc = (Ny + 1) / 2
    yc = (Nx + 1) / 2
    sx = (Ny - 1) / 2
    sy = (Nx - 1) / 2

    # Limiting the resolution to avoid excessively large output
    Nr = min(2 * Ny, max_resolution)
    Nth = min(2 * Nx, max_resolution)

    dr = (rMax - rMin) / (Nr - 1)
    dth = 2 * np.pi / Nth

    r = np.linspace(rMin, rMin + (Nr-1) * dr, Nr)
    th = np.linspace(0, (Nth-1) * dth, Nth)
    r, th = np.meshgrid(r, th)

    x = r * np.cos(th)
    y = r * np.sin(th)
    xR = x * sx + xc
    yR = y * sy + yc

    # Flatten xR and yR to 1-D arrays for interp2d
    xR_flat = xR.ravel()
    print(f"xR_flat shape is : {xR_flat.shape}")
    yR_flat = yR.ravel()
    print(f"yR_flat shape is : {yR_flat.shape}")

    try:
        # 2D interpolation
        f = interp2d(np.arange(Nx), np.arange(Ny), imC, kind='cubic')
        imP = f(xR_flat, yR_flat)
    except RuntimeError as e:
        # Handle interpolation errors
        print(f"Error during interpolation: {e}")
        return None

    # Reshape imP back to the original shape and replace NaNs with 0
    imP = imP.reshape(Nr, Nth)
    imP[np.isnan(imP)] = 0

    return imP
