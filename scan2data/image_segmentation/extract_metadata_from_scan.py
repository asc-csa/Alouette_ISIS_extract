# -*- coding: utf-8 -*-
"""
Code to extract metadata part of a raw scanned image
"""

# Library imports
import numpy as np

def extract_metadata(raw_img_array, limits_ionogram):
    """Extract metadata part of a raw scanned image and return coordinates delimiting its limits
    
    :param raw_img_array: UTF-8 grayscale 2D array of values ranging from [0,255] representing raw scanned simage
    :type raw_img_array: class: `numpy.ndarray`
    :param limits: limits delimiting the ionogram part of the raw scanned image i.e. [x_axis_left_limit ,x_axis_right_limit, y_axis_upper_limit, y_axis_lower_limit]
    :type limits: list
    :return: (type_metadata,raw_metadata) i.e. (location of metadata where {0:'left',1:'right',2:'top', 3:'bottom'} in the image,  UTF-8 grayscale 2D array of values ranging from [0,255] representing metadata part of raw scanned image)
    :rtype: (int, class: `numpy.ndarray`)
    """
    
    # Limits delimiting the ionogram part of the image
    x_axis_left_limit, x_axis_right_limit, y_axis_upper_limit, y_axis_lower_limit = limits_ionogram
    
    
    # Extract each rectangular block besides the ionogram
    rect_left = raw_img_array[:,0:x_axis_left_limit]
    rect_right = raw_img_array[:,x_axis_right_limit::]
    rect_top = raw_img_array[0:y_axis_upper_limit ,:]
    rect_bottom = raw_img_array[y_axis_lower_limit:: ,:]
    
    # Assumption: The location of the metadata will correspond to the rectangle with the highest area
    rect_list = [rect_left, rect_right, rect_top, rect_bottom]
    rect_areas = [rect.shape[0] *rect.shape[1] for rect in rect_list]
    dict_mapping_meta = {0:'left', 1:'right', 2:'top', 3:'bottom'}
    type_metadata_idx = np.argmax(rect_areas) 
    raw_metadata = rect_list[type_metadata_idx]
    type_metadata = dict_mapping_meta[type_metadata_idx ]
    
    return (type_metadata, raw_metadata)
 
    
