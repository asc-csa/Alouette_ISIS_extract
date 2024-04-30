# -*- coding: utf-8 -*-
"""
Code to trim extracted raw metadata

"""
# Library imports
import math

import numpy as np
import cv2


def connected_components_metadata_location(meta,min_count=50, max_count=1000):
    """Use the connected component algorithm to find the location of the metadata
    
    :param meta: binarized UTF-8 2D array of values (0 or 1) array containing metadata
    :type meta: class: `numpy.ndarray`
    :param min_count: minimum number of pixels to be considered metadata dot/num, defaults to 50
    :type min_count: int, optional
    :param max_count: maximum number of pixels to be considered metadata dot/num, defaults to 1000
    :type max_count: int, optional
    :return: metadata labelled by the connected component algorithm ie UTF-8 2D array of values where each value correspond to belonging to a metadata group
    :rtype: class: `numpy.ndarray`
    """
    # Run the connected component algorithm to label the metadata rectangle
    _, labelled = cv2.connectedComponents(meta)

    # Dictionary of label:counts
    unique, counts = np.unique(labelled, return_counts = True)
    dict_components = dict(zip(unique, counts ))

    # Remove outliers ie pixels not part of metadata
    dict_subset = {}
    dict_outlier = {}
    for k,v in dict_components.items():
        if v > min_count and v < max_count:
            dict_subset[k] = v
        else: 
            dict_outlier[k] = v

    if key_list_to_remove := list(dict_outlier.keys()):
        for k in key_list_to_remove:
            labelled[labelled==k] = 0

    return labelled

def grouped_limits_metadata(meta, row_or_col,
            offset_right=20,grouped=10):
    """Returns the upper and lower limits of the metadata part of the raw metadata (rectangle of scan with metadata) by row or column using mean-based thresholding
    
    :param meta: UTF-8 2D array of values  representing metadata
    :type meta: class: `numpy.ndarray`
    :param row_or_col: 0 for column or 1 for row
    :type row_or_col: int
    :param offset_right: offset (number of pixels) to avoid including ionogram, defaults to 20
    :type  offset_right: int, optional
    :param grouped: size (number of pixels) of group, defaults to 10
    :type  grouped: int, optional
    :return:  grouped*limits[0],grouped*limits[-1]] i.e. the grouped upper and lower limits of the raw metadata (rectangle of scan with metadata) by row (row_or_col=1) or column (row_or_col=0)
    :rtype: int,int
    """

    sum_values = np.sum(meta, row_or_col)
    sum_values = sum_values[:-offset_right]
    mean_values = [sum(sum_values[i:i + grouped])/(meta.shape[row_or_col]) 
                        for i in range(0,len(sum_values),grouped)]

    if math.ceil(len(mean_values) / grouped) > len(mean_values):
        mean_values = mean_values.append(sum(sum_values[i + grouped:-1])/grouped)

    normalized_mean_values = (mean_values - np.min(mean_values))/(np.max(mean_values)- np.min(mean_values))
    threshold = 0
    limits = [i for i, mean in enumerate(normalized_mean_values) if mean >threshold ]

    return grouped*limits[0],grouped*limits[-1]



def leftside_metadata_trimming(connected_meta,meta_binary,
                               offset = 20, max_width_metadata=300,threshold_mean_ionogran_chunk_left=0.01):
    
    """Metadata trimming protocol (mean-based thresholding) for metadata located on the left of ionograms
    
    :param connected_meta: metadata labelled by the connected component algorithm ie UTF-8 2D array of values where each value correspond to belonging to a metadata group
    :type connected_meta: class: `numpy.ndarray`
    :param meta: binarized UTF-8 2D array of values (0 or 1) array containing metadata
    :type meta: class: `numpy.ndarray`
    :param offset: offset (number of pixels) to avoid including ionogram, defaults to 20
    :type  offset: int, optional
    :param max_width_metadata: maximum width of trimmed rectangles with metadata, defaults to 300
    :type  max_width_metadata: int, optional
    :param threshold_mean_ionogran_chunk_left: minmum mean area to signal the presence of ionogram chunks, defaults to 0.01
    :type  max_width_metadata: int, optional
    :return: trimmed metadata i.e.  trimmed binarized UTF-8 2D array of values (0 or 1) array containing metadata
    :rtype: class: `numpy.ndarray`
    """
    
    h_raw,w_raw = meta_binary.shape
    y_axis_upper_limit_meta, y_axis_lower_limit_meta = grouped_limits_metadata(connected_meta, 1)
    x_axis_left_limit_meta ,x_axis_right_limit_meta = grouped_limits_metadata(connected_meta, 0)
    
    offset_right = offset

    while abs(x_axis_right_limit_meta - x_axis_left_limit_meta) > max_width_metadata:
        x_axis_left_limit_meta ,x_axis_right_limit_meta = grouped_limits_metadata(connected_meta, 0,offset_right)
        offset_right = offset_right + 10
        
    
    y_axis_upper_limit_meta_offset = max(y_axis_upper_limit_meta-offset,0)
    y_axis_lower_limit_meta_offset = min(y_axis_lower_limit_meta+ offset,h_raw)
    
    x_axis_left_limit_meta_offset = max(x_axis_left_limit_meta-offset,0)
    x_axis_right_limit_meta_offset = min(x_axis_right_limit_meta+ offset,w_raw)
    
    trimmed_metadata = meta_binary[y_axis_upper_limit_meta_offset:y_axis_lower_limit_meta_offset,
                                   x_axis_left_limit_meta_offset:x_axis_right_limit_meta_offset]
    
    #ionogram chunk on left
    if np.mean(np.mean(trimmed_metadata[:,0:offset+1],0)) != 0 and np.mean(np.mean(trimmed_metadata[:,offset+1:2*offset+1],0)) < threshold_mean_ionogran_chunk_left:
        x_axis_left_limit_meta ,_ = grouped_limits_metadata(trimmed_metadata[:,offset+1::], 0)
        trimmed_metadata = meta_binary[y_axis_upper_limit_meta_offset :y_axis_lower_limit_meta_offset ,
                                   x_axis_left_limit_meta :x_axis_right_limit_meta_offset ]
        
    return trimmed_metadata



def bottomside_metadata_trimming(connected_meta,opened_meta,
                                 h_window=100,w_window=700,starting_y = 0, starting_x=15,step_size=10,trim_if_small=10):

    '''Metadata trimming protocol (sliding windowing) for metadata located on the bottom of ionograms
    
    :param connected_meta: metadata labelled by the connected component algorithm ie UTF-8 2D array of values where each value correspond to belonging to a metadata group
    :type connected_meta: class: `numpy.ndarray`
    :param opened_meta:  UTF-8 2D array of values representing raw metadata after morphological operations including opening
    :type opened_meta: nclass: `numpy.ndarray`
    :param h_window: height of sliding window, defaults to 100
    :type h_window: int, optional
    :param w_window: width of sliding window, defaults to 700
    :type w_window: int, optional
    :param starting_y: by how many pixels from the top to start windowing process, defaults to 0
    :type starting_y: int, optional
    :param starting_x: by how many pixels from the left to start the windowing process, defaults to 15
    :type starting_x: int, optional
    :param step_size: by how much sliding window moves to the right and/or bottom, defaults to 10
    :type step_size: int, optional
    :param trim_if_small: by how many pixels to trim metadata's height or width if they are smaller than h_window or w_window,defaults to 11
    :type trim_if_small: int, optional
    :return: trimmed metadata i.e.  trimmed UTF-8 2D array of values containing metadata (window with highest mean area)
    :rtype: class: `numpy.ndarray`
    '''
    def sliding_window(image,starting_y,starting_x,h_window,w_window,step_size):
        '''Sliding window generator object'''
        h_img,w_img = image.shape
        for y in range(starting_y, h_img- h_window,step_size):
            for x in range(starting_x, w_img- w_window, step_size):
                yield y,x,image[y:y+h_window, x:x +w_window ]
          
    h_raw,w_raw = opened_meta.shape
    
    if h_window + step_size  >= h_raw:
        h_window = h_raw -trim_if_small
    if w_window + step_size>= w_raw:
        w_window = w_raw -trim_if_small
    
    s = sliding_window(connected_meta,starting_y,starting_x,h_window,w_window,step_size)

    max_window = connected_meta[starting_y:h_window+starting_y,
                 starting_x:w_window+starting_x ]
    max_mean = np.mean(max_window)
    y_max= starting_y
    x_max = starting_x
    for y,x,window in s:
        tmp = window
        mean = np.mean(tmp)
        if mean > max_mean:
            max_window = tmp
            max_mean  = mean
            y_max= y
            x_max =x

    trimmed_metadata =  opened_meta[y_max:y_max+h_window,x_max:x_max+w_window]

    return trimmed_metadata



def trimming_metadata(raw_metadata,type_metadata,
                     median_kernel_size=5,
                     opening_kernel_size = (3,3)):
    """Trim the rectangle containing the metadata to the smallest workable area
    
    :param raw_metadata: UTF-8 grayscale 2D array of values ranging from [0,255] representing rectangular metadata part of raw scanned image
    :type raw_metadata: class: `numpy.ndarray`
    :param type_metadata: where the detected metadata is located compared to the ionogram i.e. 'left', 'right', 'top', 'bottom'
    :type type_metadata: str
    :param median_kernel_size: size of the filter for median filtering morphological operation, defaults to 5
    :type median_kernel_size: int, optional
    :param opening_kernel_size: size of the filter for opening morphological operation, defaults to (3,3)
    :type opening_kernel_size: (int,int), optional
    :return: trimmed metadata i.e.  trimmed UTF-8 2D array of values  containing metadata 
    :rtype: class: `numpy.ndarray`
    :raises Exception: returns np.nan if there is an error
        
    """
    try:
        # Median filtering to remove salt and pepper noise
        median_filtered_meta = cv2.medianBlur(raw_metadata,median_kernel_size)

        # Opening operation: Erosion + Dilation
        kernel_opening = np.ones(opening_kernel_size,np.uint8)
        opened_meta = cv2.morphologyEx(median_filtered_meta,cv2.MORPH_OPEN,kernel_opening)

        # Binarizatinon for connected component algorithm
        _,meta_binary = cv2.threshold(opened_meta, 127,255,cv2.THRESH_BINARY)    

        # Run connected component algorithm
        connected_meta = connected_components_metadata_location(meta_binary)

        # These lines were added just for checking the extracted metadata
        # cv2.imshow("test", trimmed_metadata)
        # cv2.waitKey(0)

        return leftside_metadata_trimming(connected_meta, meta_binary) if type_metadata == 'left' else bottomside_metadata_trimming(connected_meta, meta_binary)

    except Exception:
        return np.nan
