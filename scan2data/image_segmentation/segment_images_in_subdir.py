# -*- coding: utf-8 -*-
"""
For each of the raw images in a subdirectory i.e. R014207948/1743-9/, segment it into ionogram plot and trimmed metadata while handling errors and recording loss of data
"""

# Library imports
import glob
import sys

import cv2
import pandas as pd
import numpy as np

try :
    from extract_ionogram_from_scan import extract_ionogram
    from extract_metadata_from_scan import extract_metadata
    from trim_raw_metadata import trimming_metadata
    sys.path.append('../')
    from helper_functions import record_loss

    
except ModuleNotFoundError:
    sys.path.append('../')
    from helper_functions import record_loss
    from image_segmentation.extract_ionogram_from_scan import extract_ionogram
    from image_segmentation.extract_metadata_from_scan import extract_metadata
    from image_segmentation.trim_raw_metadata import trimming_metadata

#List of subdirectories requiring geometric transformation (rotation,reflection)

LIST_FLIP_VERTICAL = ["R014207815/3508-A19",
                      "R014207954/2201-1A",
                      "R014207907F/289"]
#R014207909F/730: 0110-0205


LIST_ROTATE_180 = ["R014207962/1502-3A",
                   "R014207962/1505-1B",
                   "R014207965/1627-4A",
                   "R014207965/1647-4A",
                   "R01420796/1655-6A"]



def segment_images(subdir_location, regex_img,
                  cutoff_width = 300, cutoff_height=150,
                  min_leftside_meta_width = 50, min_bottomside_meta_height=15): #min_bottomside_meta_height=25
    """From all the raw images in a subsubdirectory, extract the ionogram and trimmed metadata while handling errors and recording loss of data
    
    :param subdir_location: full path of the subdir
    :type subdir_location: str
    :param regex_img: regular expression to extract image
    :type regex_img: str
    :param cutoff_width: the width of an ionogram should be within cutoff_width of the median width of all the ionogram in a subdirectory, defaults to 300
    :type cutoff_width: int, optional
    :param cutoff_height: the height of an ionogram should be within cutoff_height of the median height of all the ionogram in a subdirectory, defaults to 150
    :type cutoff_height: int, optional
    :param min_leftside_meta_width: the minimum width of trimmed metadata located on the left side of ionograms, defaults to 50
    :type min_leftside_meta_width: int, optional
    :param min_bottomside_meta_height: the minimum height of trimmed metadata located on the bottom side of ionograms, defaults to 25
    :type min_bottomside_meta_height: int, optional
    :return: df_img,df_loss,df_outlier i.e. dataframe containing extracted ionograms and trimmed metadata from all the images in a directory,dataframe containing file names leading to runtime errors, dataframe containing file names that do not pass pre-established filters (metadata size, ionogram size)
    :rtype: (class: `pandas.core.frame.DataFrame`,class: `pandas.core.frame.DataFrame`,class: `pandas.core.frame.DataFrame`)
    .. todo:: complete flip_vertical ie list of subdirectories requiring flipping
    """
    # List of raw image files in subdirectory
    regex_raw_image = subdir_location + regex_img
    list_images = glob.glob(regex_raw_image)
    
    # If flipping/rotating is required for all the images in the subdirectory
    path = subdir_location.replace('\\', '/')
    flip_vertical = any([subdir_dir in path for subdir_dir in LIST_FLIP_VERTICAL])
    rotate_180 = any([subdir_dir in path for subdir_dir in LIST_ROTATE_180])

    # DataFrame for processing
    df_img = pd.DataFrame(data = {'file_name': list_images})     
    
    #Read each image in an 2D UTF-8 grayscale array
    if flip_vertical == True:
        df_img['raw'] = df_img['file_name'].map(lambda file_name: cv2.flip(cv2.imread(file_name,0),1))
    else:
        df_img['raw'] = df_img['file_name'].map(lambda file_name: cv2.imread(file_name,0))
    
    # Extract ionogram and coordinates delimiting its limits
    df_img['limits'], df_img['ionogram'] = zip(*df_img['raw'].map(lambda raw_img: extract_ionogram(raw_img))) #from extract_ionogram_from_scan.py
    
    # Record the files whose ionogram extraction was not successful
    df_loss_ion_extraction, loss_ion_extraction = record_loss(df_img,'image_segmentation.extract_ionogram_from_scan.extract_ionogram',subdir_location)

    # Remove files whose ionogram extraction was not successful
    df_img = df_img[~loss_ion_extraction]
    
    if rotate_180 == True:
        df_img['ionogram'] = df_img['ionogram'].map(lambda ionogram: np.rot90(ionogram, 2))

    # Extract the shape of each ionogram in subdirectory
    df_img['height'],df_img['width'] = zip(*df_img['ionogram'].map(lambda array_pixels: array_pixels.shape))
    
    #Find median height and width of ionogram in subdirectory
    median_height = np.median(df_img['height'])
    median_width = np.median(df_img['width'])
    
    # Find and remove ionogram outliers    
    conditional_list_ionogram = [abs(df_img['height'] -median_height) > cutoff_height,abs(df_img['width'] - median_width) > cutoff_width] 
    outlier_ionogram = np.any(conditional_list_ionogram,axis = 0)
    
    df_outlier_ionogram,_ = record_loss(df_img,'image_segmentation.segment_images_in_subdir.segment_images: iono size outlier',subdir_location,['file_name','height','width'],outlier_ionogram)
    
    # Log outlier
    if not df_outlier_ionogram.empty:
        df_outlier_ionogram[ 'details'] = df_outlier_ionogram.apply(lambda row: 'height: ' + str(row['height'])+',width: ' + str(row['width']), 1)
        df_outlier_ionogram = df_outlier_ionogram[['file_name','func_name','subdir_name','details']]
    else:
        df_outlier_ionogram = df_outlier_ionogram[['file_name','func_name','subdir_name']]
    
    # Remove outlier
    df_img = df_img[~outlier_ionogram]
    

    # Raw metadata
    print(df_img.columns)
    df_img['metadata_type'], df_img['raw_metadata'] = zip(*df_img.apply(lambda row: extract_metadata(row['raw'], row['limits']), 1)) #from extract_metadata_from_scan
    if rotate_180 == True:
        df_img['raw_metadata'] = df_img['raw_metadata'].map(lambda raw_metadata: np.rot90(raw_metadata, 2))
    
    # There should be no metadata on left and top, especially after flipping
    '''outlier_metadata_location = np.any([df_img['metadata_type'] == 'right', df_img['metadata_type']=='top'], axis=0)
    df_outlier_metadata_location ,_ =  record_loss(df_img,'image_segmentation.segment_images_in_subdir.segment_images: metadata not on left or bottom',subdir_location,
                                         ['file_name','metadata_type'],outlier_metadata_location )

    if not df_outlier_metadata_location.empty:
        df_outlier_metadata_location['details'] = df_outlier_metadata_location.apply(lambda row: str(row['metadata_type']),1)
        df_outlier_metadata_location = df_outlier_metadata_location[['file_name','func_name','subdir_name','details']]
    else:
        df_outlier_metadata_location = df_outlier_metadata_location[['file_name','func_name','subdir_name']]
    
    # Remove loss from detected metadata not being on the left or bottom
    df_img = df_img[~outlier_metadata_location]'''
    
    # Trimmed metadata
    df_img['trimmed_metadata'] = df_img.apply(lambda row: trimming_metadata(row['raw_metadata'], row['metadata_type']), 1) #from trim_raw_metadata.py
    df_loss_trim,loss_trim = record_loss(df_img,'image_segmentation.trim_raw_metadata.trimming_metadata',subdir_location)

    
    # Remove files whose metadata trimming was not successful
    df_img = df_img[~loss_trim]
    
    
    # Check if metadata too small
    df_img['meta_height'],df_img['meta_width'] = zip(*df_img['trimmed_metadata'].map(lambda array_pixels: array_pixels.shape))
    outlier_size_metadata = np.logical_or(np.logical_and(df_img['metadata_type'] == 'left', 
                                                      df_img['meta_width'] < min_leftside_meta_width),
                                       np.logical_and(df_img['metadata_type'] == 'bottom', 
                                                      df_img['meta_height'] < min_bottomside_meta_height))
        
    df_outlier_metadata_size, _ = record_loss(df_img,'image_segmentation.segment_images_in_subdir.segment_images: metadata size outlier',subdir_location,
                                           ['file_name','metadata_type','meta_height','meta_width'],outlier_size_metadata)

    if not df_outlier_metadata_size.empty:
        df_outlier_metadata_size['details'] = df_outlier_metadata_size.apply(lambda row: row['metadata_type'] + '_height: ' + \
                                                    str(row['meta_height'])+',width: ' + str(row['meta_width']),1)
        df_outlier_metadata_size = df_outlier_metadata_size[['file_name','func_name','subdir_name','details']]
        
    else:
        df_outlier_metadata_size = df_outlier_metadata_size[['file_name','func_name','subdir_name']]
    
    # Remove files whose metadata too small
    df_img = df_img[~outlier_size_metadata]
    
    
    # Dataframe recording loss from programming errors
    df_loss = pd.concat([df_loss_ion_extraction,df_loss_trim])
    
    # Dataframe recording loss from various filters i.e. metadata too small, ionogram too small/big
    df_outlier = pd.concat([df_outlier_ionogram,df_outlier_metadata_size]) #df_outlier_metadata_location
    
    return df_img, df_loss, df_outlier
