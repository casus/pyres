import numpy as np

def linmap(val, valMin, valMax, mapMin=None, mapMax=None):
    """
    Performs a linear mapping of val from the range [valMin,valMax] to the range [mapMin,mapMax].

    Parameters:
        val (numpy.ndarray): Input value
        valMin (float): Minimum value of the range of val
        valMax (float): Maximum value of the range of val
        mapMin (float, optional): Minimum value of the new range of val. Defaults to valMin.
        mapMax (float, optional): Maximum value of the new range of val. Defaults to valMax.

    Returns:
        numpy.ndarray: Rescaled value.

    Example:
        rsc = linmap(val,0,255,0,1)  # map the uint8 val to the range [0,1]
    """

    # If mapMin and mapMax aren't provided, normalize the data between valMin and valMax
    if mapMin is None and mapMax is None:
        mapMin, mapMax = valMin, valMax
        valMin, valMax = np.min(val), np.max(val)

    # Convert the input value between 0 and 1
    tempVal = (val - valMin) / (valMax - valMin)

    # Clamp the value between 0 and 1
    tempVal[tempVal < 0] = 0
    tempVal[tempVal > 1] = 1

    # Rescale and return
    rsc = tempVal * (mapMax - mapMin) + mapMin

    return rsc
