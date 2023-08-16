# -*- coding: utf-8 -*-
"""
Code to determine the grid from all the ionograms in a folder 
using adjusted peak detection (From the weighed sum of all the image plots in a subsubdirectory,
the indices corresponding to the grid values are determined)

"""
# Library imports
import numpy as np
import pandas as pd
import scipy.signal as signal


#Determined from determine_default_grid_values
HZ = [1.5,2.0,2.5,3.5,4.5,5.5,6.5,7.0,7.5,8.5,9.5,10.5,11.5]
DEFAULT_HZ_COORD = [149,273,335,390,496,604,711,837,898,964,1128,1314,1444]
MEAN_HZ = [0.5*(num + DEFAULT_HZ_COORD[i+1])for i, num in enumerate(DEFAULT_HZ_COORD[:-1])]
UPPER_LIMIT_HZ_COORD =[89] + MEAN_HZ 
LOWER_LIMIT_HZ_COORD = MEAN_HZ + [1510]

KM_DEFAULT_100 = 55
KM_DEFAULT_200 = 110

def all_stack(df_img):
    """Returns the equally weighed sum of all the correctly extracted ionogram plot areas in a subsubdirectory 
    z
    :param df_img: Dataframe contaning all the correctly extracted ionogram plot areas in a subsubdirectory (output of image_segmentation.segment_images_in_subdir.segment_images)
    :type df_img: class: `pandas.core.frame.DataFrame`
    :param cutoff_width: the width of an ionogram should be within cutoff_width of the median width of all the ionogram in a subdirectory (should be the same as the one used in scan2data.image_segmentation.segment_images_in_subdir.segment_images)
    :type cutoff_width: int
    :param cutoff_height: the height of an ionogram should be within cutoff_height of the median height of all the ionogram in a subdirectory (should be the same as the one used in scan2data.image_segmentation.segment_images_in_subdir.segment_images)
    :type cutoff_height: int
    :returns: weighed_sum i.e. equally weighed sum of all the extracted ionogram plot areas in a subsubdirectory
    :rtype: class: `numpy.ndarray`
    """
    
    # Pad the image if needed
    max_h = df_img["height"].max()
    max_w =  df_img["width"].max()
    median_h = int(np.median(df_img["height"]))
    median_w = int(np.median(df_img["width"]))
    

    df_img["padded"] = df_img["ionogram"].apply(lambda img: np.pad( img, ((0,max_h-img.shape[0]),(0,max_w-img.shape[1])),mode="constant",constant_values=1))
    
    #Weighed sum of the ionograms in a subdirectory <-- Q: WHY DO WE HAVE TO TAKE THE WEIGHTED SUM OF ALL OF THE IONOGRAMS? SHOULDN'T EACH IONOGRAM BE ANALYZED INDIVIDUALLY?
    weight = len(df_img.index)
    weighed_sum = weight * np.sum((df_img["padded"]).tolist(), axis = 0)
    median_h = median_h if max_h>median_h else median_h-1

    return weighed_sum[0:-(max_h-median_h),0:-(max_w-median_w)]


def indices_highest_peaks(img, row_or_col,
                      peak_prominence_threshold=0.1, distance_between_peaks=5):
    """Determines and returns the indices of peak median values from the rows or column of an image
    
    :param img: grayscale image in the form of an 2D uint8 array
    :type img: class: `numpy.ndarray`
    :param row_or_col: 0 for colum or 1 for row
    :type row_or_col: int
    :param peak_prominence_threshold: the threshold to detect peaks that correspond to the grid lines, defaults to 0.1
    :type peak_prominence_threshold: int, optional
    :param distance_between_peaks: the minimum number of samples between subsequent peaks, defaults to 5
    :type distance_between_peaks: int, optional
    :returns: select_peaks i.e. array of the indices of peak median values from the rows or column of an image
    :rtype: class: `numpy.ndarray`
    """
    
    # Median values along each row or column
    median_values = np.median(img,row_or_col)

    # Normalize median values so they are between 0 and 1
    median_values_normalized = (median_values - np.min(median_values))/(np.max(median_values)-np.min(median_values))

    # Prepare peaks for peak detection function: the peaks should be pointing upwards
    peaks_function = 1 + -1*median_values_normalized 

    # Detect all peaks
    select_peaks, _ = signal.find_peaks(peaks_function, distance=distance_between_peaks, prominence=peak_prominence_threshold) #from scipy.signal

    #Remove edges from peaks
    h,w = img.shape
    select_peaks = select_peaks[:w] if row_or_col == 0 else select_peaks[:h]
    return select_peaks
    

def adjust_arr_peaks(weighed_sum,arr_peaks,desired_length,row_or_col,
                  distance_between_peaks=30,peak_prominence_threshold=0.1,n_tries=1000,update_amount=0.01):
    """Adjust an array of peaks to the desired length and returns it. 
    
     :param weighed_sum: equally weighed sum of all the image plot areas in a subsubdirectory 
     :type weighed_sum: class: `numpy.ndarray`
     :param arr_peaks: array of peaks to adjust to the desired length
     :type arr_peaks: class: `numpy.ndarray`
     :param desired_length: number of elements desired in array
     :type desired_length: int
     :param row_or_col: 0 for colum or 1 for row
     :type row_or_col: int
     :param distance_between_peaks: the minimum number of samples between subsequent peaks, defaults to 30
     :type distance_between_peaks: int, optional
     :param peak_prominence_threshold: the threshold to detect peaks that correspond to the grid lines, defaults to 0.1
     :type peak_prominence_threshold: int, optional
     :param n_tries: the number of maximum tries to adjust arr, defaults to 1000
     :type n_tries: int, optional
     :param update_amount: by how much peak_prominence_threshold is updated for each iteration, defaults to 0.01
     :type update_amount: int, optional
     :returns: select_peaks i.e. adjusted array of the indices of peak median values from the rows or column of an image
     :rtype: class: `numpy.ndarray`
     ..note:: To prevent infinite loops, the script only runs for a maximum of n_tries times
    """
    
    arr_peaks = indices_highest_peaks(weighed_sum, row_or_col, peak_prominence_threshold, distance_between_peaks)  #Q: WHY IS THE FUNCTION TAKING IN arr_peaks IF IT IS ALREADY CALCULATING IT HERE?

    # Adjust if lenght is not the desired length by re-running indices_highest_peaks with different parameters
    while len(arr_peaks) != desired_length and n_tries !=0:
        
        if len(arr_peaks) > desired_length:
            # increase peak_prominence_threshold 
            peak_prominence_threshold = peak_prominence_threshold + update_amount
        else:
            # decrease peak_prominence_threshold
            peak_prominence_threshold = peak_prominence_threshold - update_amount
        arr_peaks = indices_highest_peaks(weighed_sum, row_or_col,
                                           peak_prominence_threshold, distance_between_peaks)
        n_tries = n_tries - 1


    return arr_peaks


def get_grid_mappings(weighed_sum,
                      use_defaults=True,min_index_row_peaks=40):
    """Determines and returns the the mapping between coordinate values and frequency/depth values in a subdirectory
    
    :param weighed_sum: equally weighed sum of all the image plot areas in a subsubdirectory 
    :type weighed_sum: class: `numpy.ndarray`
    :param use_defaults: use default values , defaults to True
    :type use_defaults: bool
    :param min_index_row_peaks: starting index to consider for peaks to determine km lines,defaults to 40
    :type min_index_row_peaks: int, optional
    :returns:  col_peaks,row_peaks,mapping_Hz, mapping_km i.e. one-dimmensional array of detected peaks of ionogram by column, one-dimmensional array of detected  peaks of by row, dictionary mapping of depth (km) to y coordinates  , dictionary mapping of frequency (Hz) to x coordinates
    :rtype: class: `numpy.ndarray`,class: `numpy.ndarray`,class: `dict`, class: `dict`
    :raises Exception: returns np.nan,np.nan,np.nan,np.nan
    """
    # Detect peaks
    col_peaks = indices_highest_peaks(weighed_sum, 0)
    row_peaks = indices_highest_peaks(weighed_sum, 1)

    # Map col_peaks to Hz values
    if len(col_peaks) == len(HZ):
        mapping_Hz = dict(zip(HZ,col_peaks)) 

    # Map adjusted col_peaks to Hz values
    else:
        try: 
            col_peaks = adjust_arr_peaks(weighed_sum, col_peaks, len(HZ), 0)
            mapping_Hz = dict(zip(HZ,col_peaks)) 

            # Map adjusted HZ values to default coordinates if need be
            if use_defaults:
                for i,key in enumerate(HZ):
                    if mapping_Hz[key] > UPPER_LIMIT_HZ_COORD[i] or mapping_Hz[key] < LOWER_LIMIT_HZ_COORD[i]:
                        mapping_Hz[key] = DEFAULT_HZ_COORD[i]
        except:
            if use_defaults:
                 mapping_Hz = dict(zip(HZ,DEFAULT_HZ_COORD)) 
            else:    
                return np.nan,np.nan,np.nan,np.nan


    row_peaks = row_peaks[row_peaks > min_index_row_peaks]

    try:
        row_100 = row_peaks[0] #Should be around 30 for 100 km
        row_200 = row_peaks[1] #Should be around 30 for 100 km
    except:
        if not use_defaults:
            return np.nan,np.nan,np.nan,np.nan

        row_100 = KM_DEFAULT_100
        row_200 = KM_DEFAULT_200
    if use_defaults:
        if abs(row_100 - KM_DEFAULT_100) > abs(KM_DEFAULT_200 - KM_DEFAULT_100):
            row_100 = KM_DEFAULT_100
        if abs(row_200 - KM_DEFAULT_200) > abs(KM_DEFAULT_200 - KM_DEFAULT_100):
            row_200 = KM_DEFAULT_200   

    mapping_km = {100:row_100,200:row_200}

    return col_peaks, row_peaks, mapping_Hz, mapping_km
