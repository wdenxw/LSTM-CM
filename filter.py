"""Filter
the median filter

Args:
    data of nparray to filter

Returns:
    return the filtered data

Raises:
    None
"""

import numpy as np
import scipy.signal as signal



def median_filter(array):#
    newdata = []
    for i in range(array.shape[1]):
        data_transfer = (array[:, i])
        new=signal.medfilt(data_transfer, 5)
        newdata.append(new)
    return np.array(newdata).T