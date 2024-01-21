import numpy as np

def pop_from_array(arr):
    """Pops the last element from a NumPy array and returns the element and the new array."""
    popped_element = arr[-1]
    new_arr = arr[:-1]
    return popped_element, new_arr

def getDcorrLocalMax(d):
    Nr = len(d)
    if Nr < 2:
        ind = 0
        A = d[0]
    else:
        # Find the maximum of d
        A, ind = np.max(d), np.argmax(d)
        while len(d) > 1:
            if ind == len(d) - 1:
                _, d = pop_from_array(d)
                A, ind = np.max(d), np.argmax(d)
            elif ind == 0:
                break
            elif (A - np.min(d[ind:])) >= 0.0005:
                break
            else:
                _, d = pop_from_array(d)
                A, ind = np.max(d), np.argmax(d)
        if not ind:
            ind = 0
            A = d[0]

    return ind, A








