# -*- coding: utf-8 -*-
"""
Code to extract ionogram part of a raw scanned image
"""
# Library imports
import numpy as np
import cv2

def limits_ionogram(raw_img, row_or_col,
                    starting_index_col=15):
    """Returns the upper and lower limits of the ionogram part of the scan by row or column using mean-based thresholding
    
    :param starting_img: UTF-8 grayscale 2D array of values ranging from [0,255] representing raw scanned image
    :type starting_img: class: `numpy.ndarray`
    :param row_or_col: 0 for column or 1 for row
    :type row_or_col: int
    :param starting_index_col: where the ionogram starting column should be after to protect against cuts, defaults to 15
    :type starting_index_col: int, optional
    :return:  limits[0],limits[-1] i.e. the upper and lower limits of the ionogram part of the scan by row (row_or_col=1) or column (row_or_col=0)
    :rtype: int,int
            
            
    """
    
    # Mean pixel values by by row/col
    mean_values = np.mean(raw_img, row_or_col)
    
    # Normalized mean values
    normalized_mean_values = (mean_values - np.min(mean_values))/np.max(mean_values)
    
    # Threshold is the overall mean value of the entire image
    threshold = np.mean(normalized_mean_values)
    
    if row_or_col == 0:
        #Protect against scans that includes cuts from another ionogram ex:R014207956\2394-1B\51.png 
        limits = [i for i, mean in enumerate(normalized_mean_values) if mean >threshold and i > starting_index_col]
    else:
        limits = [i for i, mean in enumerate(normalized_mean_values) if mean >threshold]
        
    return limits[0],limits[-1]


def extract_ionogram(raw_img_array):
    """Extract ionogram part of a raw scanned image and return coordinates delimiting its limits
    
    :param raw_img_array: UTF-8 grayscale 2D array of values ranging from [0,255] representing raw scanned image
    :type raw_img_array: class: `numpy.ndarray`
    :return: (limits, ionogram) i.e. (list of coordinates delimiting the limits of the ionogram part of a raw scanned image formatted as [x_axis_left_limit ,x_axis_right_limit, y_axis_upper_limit, y_axis_lower_limit], UTF-8 grayscale 2D array of values ranging from [0,255] representing ionogram part of scanned image)
    :rtype: (list,numpy.ndarray)
    :raises Exception: returns [],np.nan if there is an error
        
    """
    try:
        # Extract coordinate delimiting the ionogram part of the scan
        x_axis_left_limit ,x_axis_right_limit = limits_ionogram(raw_img_array, 0)
        y_axis_upper_limit, y_axis_lower_limit = limits_ionogram(raw_img_array, 1)

        # Extract ionogram part
        ionogram = raw_img_array[y_axis_upper_limit:y_axis_lower_limit,x_axis_left_limit:x_axis_right_limit]

        #Just added for checking the metadata part of image
        imgMetadataPart = raw_img_array[y_axis_upper_limit:y_axis_lower_limit, 15:x_axis_left_limit - 1]
        # cv2.imshow("test Metadata", imgMetadataPart)
        # cv2.waitKey(0)

        limits = [x_axis_left_limit ,x_axis_right_limit, y_axis_upper_limit, y_axis_lower_limit]
        return (limits, ionogram)

    except Exception:
        return ([],np.nan)