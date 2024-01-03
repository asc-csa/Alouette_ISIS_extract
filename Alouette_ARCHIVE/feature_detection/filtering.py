import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter
import math
from itertools import compress

from utils import find_nearest

#Resample the data to get relatively evenly spaced points
#Input is ionogram x, y, and desired delta_x(Hz) interval for resampling.
#Returns resampled x, y as lists
def resample(x, y, delta_x=0.01, max_delta_y = 400):
    #Create x array with desired spacing
    x_desired = np.arange(0, 14, delta_x)

    x_resampled = [0]
    y_resampled = [0]

    #Find value in x closest to each value in x_desired, then resample the y accordingly
    for val in x_desired:
        nearest_val, nearest_index = find_nearest(x, val)
        if nearest_val not in x_resampled:
            x_resampled.append(nearest_val)
            y_resampled.append(y[nearest_index])

    custom_filter(x_resampled, y_resampled)

    return x_resampled, y_resampled


#Apply our custom filters on the x and y trace. The input trace should already be resampled.
#Input is resampled trace as a list
#Output is list
def custom_filter(x, y, delta_y = 50):
    x_avg = np.average(x)
    to_delete = []

    _, temp_i = find_nearest(x, x_avg/1.5)
    y_prev = np.average( y[temp_i-2: temp_i+2] )  #Initializing y_prev

    #Simple filter based on difference with previous value
    for i, y_val in enumerate(y):
        # Remove points that have x > x_avg/2 and where the difference in values between sequential y is bigger than chosen delta_y value
        if abs(y_val - y_prev) > delta_y and x[i] > x_avg/1.5:
            to_delete.append(i)
        else:
            y_prev = y_val

    to_delete.reverse()
    for i in to_delete:
        del x[i]
        del y[i]

    return
