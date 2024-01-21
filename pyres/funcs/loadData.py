import numpy as np
from PIL import Image, ImageSequence
import os
from tkinter import filedialog
from tkinter import Tk

def load_data(path=None):
    """
    Generic function to load arbitrary image stacks.

    Parameters:
        path (str, optional): Full path of the file. If not provided, a file dialog will open.

    Returns:
        numpy.ndarray: Loaded image.
        str: Image full path.
    """
    if not path:
        root = Tk()
        root.withdraw()  # Hide the main window
        path = filedialog.askopenfilename()
        if not path:  # If no file was selected
            return None, None

    with Image.open(path) as img:
        frames = [np.array(frame) for frame in ImageSequence.Iterator(img)]
        im = np.stack(frames, axis=-1)
    
    print("Finished reading...")
    print(f"Stack size: {im.shape}")
    print(f"shape is : {im.shape}")
    return im


