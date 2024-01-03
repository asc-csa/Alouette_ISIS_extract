import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter
import math
from itertools import compress

#Convert from pixel coordinates to Hz, km
#Input is list or np.array
#Output is np.array
def convert_pixel_to_mapping(x_pixel, y_pixel, mapping_Hz, mapping_km):
    if not isinstance(x_pixel, np.ndarray):
        x_pixel = np.array(x_pixel)
    if not isinstance(y_pixel, np.ndarray):
        y_pixel = np.array(y_pixel)

    index_values_pixels, pixel_values = list(zip(*list(mapping_km.items())))
    multiplier = (pixel_values[1] - pixel_values[0]) / (index_values_pixels[1] - index_values_pixels[0])

    y_km = (y_pixel - pixel_values[0]) / multiplier + index_values_pixels[0]

    col_peaks = np.array(list(mapping_Hz.values()))

    x_km = []
    to_delete = []  # Tracks the index of points to delete
    # reverse mapping_km mappings
    mapping_Hz_reversed = {mapping_Hz[freq_key]: freq_key for freq_key in mapping_Hz}
    for i, x_val in enumerate(x_pixel):
        if int(x_val) in col_peaks:  # If x is on the grid line
            x_km.append(0)
        elif x_val < col_peaks.min() or x_val > col_peaks.max():  # If value is less than min frequency or more than max frequency, get rid of it
            to_delete.append(i)
        else:
            # find the 2 closest values and linearly interpolate from there
            leftmost_val = col_peaks[col_peaks < x_val].max()
            rightmost_val = col_peaks[col_peaks > x_val].min()
            multiplier = (mapping_Hz_reversed[rightmost_val] - mapping_Hz_reversed[leftmost_val]) / (rightmost_val - leftmost_val)
            x_val_km= mapping_Hz_reversed[leftmost_val] + multiplier * (x_val - leftmost_val)
            x_km.append(x_val_km)

    y_km = np.delete(y_km, to_delete)

    return np.array(x_km), y_km

#Change the mapped coordinates back to pixel values to plot on top of the ionogram image
def convert_mapping_to_pixel(x_Hz, y_km, mapping_Hz, mapping_km):
    if not isinstance(x_Hz, np.ndarray):
        x_Hz = np.array(x_Hz)
    if not isinstance(y_km, np.ndarray):
        y_km = np.array(y_km)

    km_values, index_values_km = list(zip(*list(mapping_km.items())))
    multiplier = (km_values[1] - km_values[0])/(index_values_km[1] -index_values_km[0] )

    #Get the pixel values for the corresponding km
    y_pixel = (y_km - km_values[0]) / multiplier + index_values_km[0]

    col_peaks_Hz = np.array(list(mapping_Hz.keys()))  # use the modified col_peaks ie the one with exactly 13 values
    x_pixel = []
    to_delete = []

    if x_Hz.size == 1:  # If we pass in a single point
        leftmost_val = col_peaks_Hz[col_peaks_Hz < x_Hz].max()
        rightmost_val = col_peaks_Hz[col_peaks_Hz > x_Hz].min()
        multiplier = (mapping_Hz[rightmost_val] - mapping_Hz[leftmost_val]) / (rightmost_val - leftmost_val)
        x_pixel = mapping_Hz[leftmost_val] + multiplier * (x_Hz - leftmost_val)

        return x_pixel, y_pixel

    else:  # If there are more than one points
        for i, x_val in enumerate(x_Hz):
            if x_val in col_peaks_Hz:  # On the grid line
                x_pixel.append(x_val)
            elif x_val < col_peaks_Hz.min() or x_val > col_peaks_Hz.max():
                to_delete.append(i)
            else:
                leftmost_val = col_peaks_Hz[ col_peaks_Hz < x_val ].max()
                rightmost_val = col_peaks_Hz[ col_peaks_Hz > x_val ].min()
                multiplier = (mapping_Hz[rightmost_val] - mapping_Hz[leftmost_val]) / (rightmost_val - leftmost_val)
                x_val_pixel = mapping_Hz[leftmost_val] + multiplier * (x_val - leftmost_val)
                x_pixel.append(x_val_pixel)

    y_pixel = np.delete(y_pixel, to_delete)

    return np.array(x_pixel), y_pixel
