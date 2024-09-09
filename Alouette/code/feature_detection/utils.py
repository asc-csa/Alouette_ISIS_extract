import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter
import math
from itertools import compress

#Find nearest value in numpy array
#Returns nearest value and associated index
def find_nearest(array, value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return array[idx-1], idx-1
    else:
        return array[idx], idx


#Make scatter plot for processed ionogram data
def make_plot(x, y, title, description, color='purple', s=0.05, ylim=[0, 1400]):
    plt.plot(x, y, s=s, c='purple')
    plt.title(title + ' (' + description + ')')
    plt.ylim(ylim[0], ylim[1])
    plt.gca().invert_yaxis()


#Plots the png of the ionogram from the 2D array raw_coord
def plot_ionogram_image(file_name, limits, title, description):
    try:
        im = plt.imread(file_name)
        implot = plt.imshow( im[ limits[2]:limits[3], limits[0]:limits[1] ], cmap='gray')
        plt.title(title + ' (' + description + ')')
    except:
        print('Missing file: ' + row['file_name'])


#Get the point where the ionogram reachces the max depth (Corresponds to the dip in the trace). Returns the coordinates of the point in Hz, km
def get_max_depth(x, y):
    x_avg = np.average(x)

    x_above_avg = list(compress(x, x>x_avg)) # Keep only the points above the average frequency
    y_above_avg = list(compress(y, x>x_avg))
    y_max_depth = np.nanmax(y_above_avg)
    x_max_depth = x_above_avg[ y_above_avg.index(np.nanmax(y_above_avg)) ]

    return x_max_depth, y_max_depth