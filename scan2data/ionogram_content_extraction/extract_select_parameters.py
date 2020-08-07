# -*- coding: utf-8 -*-
"""
Code to extract select parameters from extracted ionogram trace
"""

# Library imports
import sys
import numpy as np

sys.path.append('../')
from ionogram_grid_determination.grid_mapping import KM_DEFAULT_100

def extract_fmin_and_max_depth(arr_adjusted_coord,min_depth=50,if_raw=False):
    """Extract the minimum detected frequency value and maximum detected depth
    
    :param arr_raw_coord:  one-dimmensional array of values of all the pixels corresponding to the ionogram trace
    :type arr_raw_coord: class: `numpy.ndarray`
    :param min_depth: minimum depth in km to be considered, defaults to 30
    :type min_depth: int, optional
    :param if_raw: if (x,y) rather than (Hz,km) coordinates are used, defaults to False
    :type if_raw: bool, optional
    :returns: fmin, depth_max i.e. minimum frequency detected and maximum depth detected
    :rtype: float, float
    :raises Exception: returns np.nan, np.nan
    
    """
    try:
        
        adjusted_x, adjusted_y = zip(*arr_adjusted_coord)
        adjusted_x = np.array(adjusted_x)
        adjusted_y = np.array(adjusted_y)
        
        if if_raw:
            min_depth= int(min_depth * KM_DEFAULT_100/100)
        
        thresholded = adjusted_y > min_depth
        adjusted_x_thresholded = adjusted_x[thresholded]
        adjusted_y_thresholded = adjusted_y[thresholded]

        fmin = min(adjusted_x_thresholded)
        depth_max =  max(adjusted_y_thresholded)
        return fmin, depth_max
    except:
        return np.nan, np.nan


