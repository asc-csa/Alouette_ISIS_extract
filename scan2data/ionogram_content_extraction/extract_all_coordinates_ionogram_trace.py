# -*- coding: utf-8 -*-
"""
Code to extract coordinates of ionogram trace (black part)
"""

# Library imports
import sys

import cv2
import numpy as np

try:
    from extract_select_parameters import extract_fmin_and_max_depth

    sys.path.append('../')
    from helper_functions import record_loss

except ModuleNotFoundError:
    sys.path.append('../')
    from helper_functions import record_loss
    from ionogram_content_extraction.extract_select_parameters import extract_fmin_and_max_depth


def extract_ionogram_windows(binary_iono,
                             stepSize=25, windowSize=(100, 100)):
    """Clean ionogram by using small thresholding windows

    :param binary_iono: two-dimmensional uint8 array representing ionogram where the extracted threshold values are in while (1s) while the rest is in black (0s)
    :type binary_iono: class: `numpy.ndarray`
    :param stepSize: By how much window moves to the right and/or bottom
    :type stepSize: int
    :param windowSize: (height, width) of moving window
    :type windowSize: tuple
    :returns new_iono: cleaned ionogram represented by  two-dimmensional uint8 array
    :rtype:  class: `numpy.ndarray`

    """
    # TODOs: impove thresholding for windowing
    threshold = np.mean(binary_iono) * 2

    h_iono, w_iono = binary_iono.shape
    h_window, w_window = windowSize
    new_iono = np.zeros((h_iono, w_iono))

    for y in range(0, h_iono - h_window, stepSize):
        for x in range(0, w_iono - w_window, stepSize):
            box = binary_iono[y:y + h_window, x:x + w_window]
            if np.mean(box) > threshold:
                new_iono[y:y + h_window, x:x + w_window] = box

    return new_iono


def background_substraction(raw_iono):
    """Use Gaussian Mixture-based Background/Foreground Segmentation Algorithm to clean the raw ionogram

    :param iono: two-dimmensional uint8 array representing raw ionogram
    :type iono: class: `numpy.ndarray`
    :returns: two-dimmensional uint8 array representing cleaned ionogram
    :rtype class: `numpy.ndarray`

    """

    background_substracter = cv2.createBackgroundSubtractorMOG2()
    masked_iono = background_substracter.apply(raw_iono)

    return 255 - masked_iono


def extract_coord(iono, col_peaks, row_peaks,
                  threshold=200, kernel_size_blurring=5):
    """Extract (x,y) of all the pixels corresponding to the ionogram trace

    :param iono: two-dimmensional uint8 array representing raw ionogram
    :type iono: class: `numpy.ndarray`
    :param col_peaks: one-dimmensional array of detected peaks of ionogram by column
    :type col_peaks: class: `numpy.ndarray`
    :param row_peak: one-dimmensional array of detected  peaks of by row
    :type row_peaks: class: `numpy.ndarray`
    :param threshold: threshold of inverted pixel value to be considered ionogram data, defaults to 200
    :type threshold: int, optional
    :param kernel_size_blurring: kernel size for median filtering operation, defaults to 5
    :type kernel_size_blurring: int, optional
    :returns: arr_raw_coord0 ,arr_raw_coord: one-dimmensional array of (x,y) coordinates of all the pixels corresponding to the ionogram trace
    :rtype: class: `numpy.ndarray`,class: `numpy.ndarray`
    :raises Exception: returns np.nan,np.nan if there is an error

    """

    # Shape of image
    try:
        h, w = iono.shape

        # Median blurring to remove salt and pepper noise
        median_filtered_iono = cv2.medianBlur(iono, kernel_size_blurring)

        # Invert image
        inverted_iono = 255 - median_filtered_iono

        # Correct image for grid ie remove the grid
        grid = np.ones((h, w), np.uint8)
        for i in col_peaks:
            cv2.line(grid, (i, 0), (i, h), 0, 5, 1)
        for i in row_peaks:
            cv2.line(grid, (0, i), (w, i), 0, 5, 1)
        corrected_iono = np.multiply(grid, inverted_iono)

        # Assuming trace is going to be black ie mostly values close to 0 in the array
        # Thus, the inverted trace is going to be white ie values most close to 252
        # Threshold the image
        _, thresholded_iono = cv2.threshold(corrected_iono, threshold, 1, cv2.THRESH_BINARY)

        # Corrected ionogram by windowing operations
        windowed = extract_ionogram_windows(thresholded_iono)

        # y and x coordinates
        arr_y, arr_x = np.where(thresholded_iono == 1)
        arr_raw_coord0 = np.array(list(zip(arr_x, arr_y)), dtype=np.float64)

        arr_y, arr_x = np.where(windowed == 1)
        arr_raw_coord = np.array(list(zip(arr_x, arr_y)), dtype=np.float64)

        return arr_raw_coord0, arr_raw_coord  # raw_coord, windowed_coord    `
    except:
        return np.nan, np.nan


def map_coordinates_positions_to_values(arr_raw_coord, col_peaks, row_peaks, mapping_Hz, mapping_km):
    """Map (x,y) position coordinates of ionogram pixels to (Hz,km) values

    :param arr_raw_coord:  one-dimmensional array of (x,y) coordinates of all the pixels corresponding to the ionogram trace
    :type arr_raw_coord: class: `numpy.ndarray
    :param col_peaks: one-dimmensional array of detected peaks of ionogram by column
    :type col_peaks: class: `numpy.ndarray`
    :param row_peak: one-dimmensional array of detected  peaks of by row
    :type row_peaks: class: `numpy.ndarray`
    :param mapping_Hz: dictionary mapping of frequency (Hz) to x coordinates
    :type mapping_Hz: class: `dict`
    :param mapping_km:  dictionary mapping of depth (km) to y coordinates
    :type mapping_km: class: `dict`
    :returns: arr_adjusted_coord: one-dimmensional array of (Hz,km) values of all the pixels corresponding to the ionogram trace
    :rtype: class: `numpy.ndarray`
    """

    # check if there are any coordinate values recorded for the ionogram
    if len(arr_raw_coord)==0:
        return arr_raw_coord

    # remove outliers ie coordinates less coordinates corresponding to 0.5 Hz or more than corresponding to 11.5 Hz
    col_peaks = np.array(list(mapping_Hz.values()))  # use the modified col_peaks ie the one with exactly 13 values

    mask = np.logical_or(arr_raw_coord[:, 0] < col_peaks.min(), arr_raw_coord[:, 0] > col_peaks.max())
    arr_raw_coord = arr_raw_coord[~mask, :]

    # map (y,x) to (km, Hz)
    km_values, index_values_km = list(zip(*list(mapping_km.items())))
    multiplier = (km_values[1] - km_values[0]) / (index_values_km[1] - index_values_km[0])

    arr_adjusted_coord = arr_raw_coord.copy()
    arr_adjusted_coord[:, 1] = km_values[0] + (arr_adjusted_coord[:, 1] - index_values_km[0]) * multiplier

    # reverse mapping_km mappings
    mapping_Hz_reversed = {mapping_Hz[freq_key]: freq_key for freq_key in mapping_Hz}
    arr_adjusted_x = np.array([])
    for coord_x in arr_adjusted_coord[:, 0]:
        if coord_x in col_peaks:
            new_coord_x = mapping_Hz_reversed[coord_x]
        else:
            # find the 2 closest values and linearly interpolate from there
            leftmost_val = col_peaks[col_peaks < coord_x].max()
            rightmost_val = col_peaks[col_peaks > coord_x].min()
            multiplier = (mapping_Hz_reversed[rightmost_val] - mapping_Hz_reversed[leftmost_val]) / (
                        rightmost_val - leftmost_val)
            new_coord_x = mapping_Hz_reversed[leftmost_val] + multiplier * (coord_x - leftmost_val)

        arr_adjusted_x = np.append(arr_adjusted_x, new_coord_x)
    arr_adjusted_coord[:, 0] = arr_adjusted_x

    return arr_adjusted_coord


def extract_coord_subdir_and_param(df_img, subdir_location, col_peaks, row_peaks, mapping_Hz, mapping_km):
    """Extract the raw, windowed coordinates in all the raw extracted ionograms from a subdirectory, map those coordinates into (Hz, km) and extract select parameterd

    :param df_img: Dataframe containing all the correctly extracted ionogram plot areas in a subsubdirectory (output of image_segmentation.segment_images_in_subdir.segment_images)
    :type df_img: class: `pandas.core.frame.DataFrame`
    :param subdir_location: Path of the subdir_location
    :type subdir_location: string
    :param col_peaks: one-dimmensional array of detected peaks of ionogram by column
    :type col_peaks: class: `numpy.ndarray`
    :param row_peak: one-dimmensional array of detected  peaks of by row
    :type row_peaks: class: `numpy.ndarray`
    :param mapping_Hz: dictionary mapping of frequency (Hz) to x coordinates
    :type mapping_Hz: class: `dict`
    :param mapping_km:  dictionary mapping of depth (km) to y coordinates
    :type mapping_km: class: `dict`
    :returns: df_img, df_loss: i.e.  i.e. dataframe containing extracted ionogram trace coordinates from all the extracted raw ionograms in a directory,dataframe containing file names leading to runtime errors
    :rtype: class: `pandas.core.frame.DataFrame`, class: `pandas.core.frame.DataFrame`

    """
    # Get (x,y) coordinates of trace
    if(len(df_img)==1):
        c1, c2 = zip(*df_img['ionogram'].map(lambda iono: extract_coord(iono, col_peaks, row_peaks)))
        df_img['raw_coord'] = list(c1)
        df_img['window_coord'] = list(c2)
        
    else :
        df_img['raw_coord'], df_img['window_coord'] = zip(
            *df_img['ionogram'].map(lambda iono: extract_coord(iono, col_peaks, row_peaks)))

    # Remove loss
    df_loss_coord, loss_coord = record_loss(df_img,
                                            'ionogram_content_extraction.extract_all_coordinates_ionogram_trace.extract_coord_subdir',
                                            subdir_location) #from helper functions.py
    df_img = df_img[~loss_coord]

    # df_img.to_csv("U:/alouette-scanned-ionograms-processing/df_img.csv", index=False)

    # (Hz, km) coordinates
    df_img['mapped_coord'] = df_img['window_coord'].map(
        lambda windowed: map_coordinates_positions_to_values(windowed, col_peaks, row_peaks, mapping_Hz, mapping_km))

    # Select parameters extracted
    df_img['fmin'], df_img['max_depth'] = zip(
        *df_img['mapped_coord'].map(lambda mapped_coord: extract_fmin_and_max_depth(mapped_coord))) #from extract_select_parameters.py  #Q: I wonder if f_min and depth_max is actually scientifically useful? Why don't we extract mapped_coord instead?

    # Remove loss.
    df_loss_param, loss_param = record_loss(df_img,
                                            'ionogram_content_extraction.extract_select_parameters.extract_fmin_and_max_depth',
                                            subdir_location)
    df_img = df_img[~loss_param]

    df_loss = df_loss_coord.append(df_loss_param)

    return df_img, df_loss
