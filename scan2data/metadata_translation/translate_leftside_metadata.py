# -*- coding: utf-8 -*-
"""
Translate metadata (dot or numbers) located on the left of ionograms
"""

# Library imports
from itertools import chain
import sys

import cv2
import pandas as pd
import numpy as np

try:
    from leftside_metadata_grid_mapping import extract_centroids_and_determine_type,get_leftside_metadata_grid_mapping
    sys.path.append('../')
    from helper_functions import  record_loss

except ModuleNotFoundError:
    sys.path.append('../')
    from helper_functions import  record_loss
    from metadata_translation.leftside_metadata_grid_mapping import extract_centroids_and_determine_type,get_leftside_metadata_grid_mapping

LABELS_DICT = ['dict_cat_dot','dict_num_dot','dict_cat_digit','dict_num_digit']

def map_coord_to_metadata(list_cat_coord, list_num_coord, dict_mapping_cat, dict_mapping_num):
    """Map coordinate of metadata centroids to information
    
    :param list_cat_coord: list of metadata positions to map to categories   
    :type list_cat_coord: list
    :param list_num_coord: list of metadata positions to map to numbers 
    :type list_num_coord: list
    :param dict_mapping_cat: dictionary used to map coordinate positions to categories
    :type dict_mapping_cat: dict
    :param dict_mapping_num: dictionary used to map coordinate positions to numbers
    :type dict_mapping_num: dict
    :returns: dict_metadata
    :rtype: dict
    
    """
    
    try:
        
        list_coord = zip(list_cat_coord,list_num_coord)
        coord_mapping_cat = dict_mapping_cat.keys()
        coord_mapping_num = dict_mapping_num.keys()
        
        dict_metadata={}
        for cat_coord, num_coord in list_coord:
            cat_key = min(coord_mapping_cat, key=lambda x:abs(x-cat_coord))
            num_key = min(coord_mapping_num, key=lambda x:abs(x-num_coord))
            
            cat = dict_mapping_cat[cat_key]
            num = dict_mapping_num[num_key]
            
            # TODO: improve for many num
            if cat in dict_metadata:
                dict_metadata[cat].append(num)
            else:
                dict_metadata[cat] = [num]
        
        return dict_metadata
    except:
        return np.nan
    

        
def get_leftside_metadata(df_img,subdir_location,
                                dilation_kernel_size = (1,1)):
    """Reads metadata located on the left of ionograms, whether they are of type dot or type num
    
    :param df_img: Dataframe containing all the correctly extracted ionogram plot areas in a subsubdirectory (output of image_segmentation.segment_images_in_subdir.segment_images)
    :type df_img: class: `pandas.core.frame.DataFrame`
    :param subdir_location: full path of the subdir
    :type subdir_location: str
    :param dilation_kernel_size: size of the filter for dilation morphological operation, defaults to (1,1)
    :type dilation_kernel_size: tuple, optional
    :returns: df_img, df_loss,dict_mapping, dict_hist i.e. df_img containing additional information including the translated metadata, dataframe containing file names leading to runtime errors,dictionary of dictionaries where each dictionary correspond to a mapping between coordinates on the image and metadata labels, dictionary of histograms used to generated each dictionary in all_dict_mapping
    :rtype: class: `pandas.core.frame.DataFrame`, class: `pandas.core.frame.DataFrame`, dict, dict
    """

    # Centroids extraction
    df_img['rotated_metadata'] = df_img['trimmed_metadata'].map(lambda trimmed_meta: np.rot90(trimmed_meta,-1))
    #Just to check the result of the line above:
    # cv2.imshow("test", df_img['rotated_metadata'][0])
    # cv2.waitKey(0)

    kernel_dilation = np.ones(dilation_kernel_size,np.uint8)

    df_img['dilated_metadata'] = df_img['rotated_metadata'].map(lambda rotated_meta: cv2.dilate(rotated_meta,kernel_dilation ))
    #just to check the result of the line above:
    # cv2.imshow("test1", df_img['dilated_metadata'][0])
    # cv2.waitKey(0)

    df_img['x_centroids'],df_img['y_centroids'],df_img['is_dot'] = zip(*df_img.apply(lambda row: extract_centroids_and_determine_type(row['dilated_metadata'],row['file_name']),1))
    df_loss_centroids_extraction,loss_centroids_extraction = record_loss(df_img,'metadata_translation.determine_leftside_metadata_grid_mapping.extract_centroids_and_determine_type',subdir_location)
      
    # Remove files whose centroid metadata extraction was not successful
    df_img = df_img[~loss_centroids_extraction]
    
    # Determine metadata mapping for dot-type metadata and num-type metadata
    df_dot_subset = df_img[np.array(df_img['is_dot'])]
    df_num_subset = df_img[np.invert(np.array(df_img['is_dot']))]

    list_x_dot, list_y_dot,list_x_digit,list_y_digit = [0],[0],[0],[0]
    if not df_dot_subset.empty:
        list_x_dot = list(chain(*df_dot_subset['x_centroids'].tolist()))
        list_y_dot = list(chain(*df_dot_subset['y_centroids'].tolist()))
    
    if not df_num_subset.empty:
        list_x_digit = list(chain(*df_num_subset['x_centroids'].tolist()))
        list_y_digit = list(chain(*df_num_subset['y_centroids'].tolist()))
    dict_mapping,dict_hist = get_leftside_metadata_grid_mapping(list_x_dot,list_y_dot,list_x_digit,list_y_digit,subdir_location)

    # Determine the value of metadata based on the mappings
    df_img['dict_metadata'] = 'empty'
    df_img['dict_metadata'] = df_img.apply(lambda row: 
        map_coord_to_metadata(row['x_centroids'],row['y_centroids'],dict_mapping['dict_cat_dot'], dict_mapping['dict_num_dot']) if row['is_dot'] 
        else map_coord_to_metadata(row['x_centroids'],row['y_centroids'],dict_mapping['dict_cat_digit'], dict_mapping['dict_num_digit']),1)
    df_loss_mapping,loss_mapping = record_loss(df_img,'metadata_translation.translate_leftside_metadata.map_coord_to_metadata',subdir_location)
    df_img = df_img[~loss_mapping]
    
    df_loss = pd.concat([df_loss_centroids_extraction,df_loss_mapping])
    
    return df_img, df_loss,dict_mapping, dict_hist


def get_bottomside_metadata(df_img, subdir_location,
                            dilation_kernel_size=(1, 1)):
    """Reads metadata located on the botton of ionograms

    :param df_img: Dataframe containing all the correctly extracted ionogram plot areas in a subsubdirectory (output of image_segmentation.segment_images_in_subdir.segment_images)
    :type df_img: class: `pandas.core.frame.DataFrame`
    :param subdir_location: full path of the subdir
    :type subdir_location: str
    :param dilation_kernel_size: size of the filter for dilation morphological operation, defaults to (1,1)
    :type dilation_kernel_size: tuple, optional
    :returns: df_img, df_loss,dict_mapping, dict_hist i.e. df_img containing additional information including the translated metadata, dataframe containing file names leading to runtime errors,dictionary of dictionaries where each dictionary correspond to a mapping between coordinates on the image and metadata labels, dictionary of histograms used to generated each dictionary in all_dict_mapping
    :rtype: class: `pandas.core.frame.DataFrame`, class: `pandas.core.frame.DataFrame`, dict, dict
    """

    # Centroids extraction
    # df_img['rotated_metadata'] = df_img['trimmed_metadata'].map(lambda trimmed_meta: np.rot90(trimmed_meta,-1))
    kernel_dilation = np.ones(dilation_kernel_size, np.uint8)
    df_img['dilated_metadata'] = df_img['trimmed_metadata'].map(
        lambda trimmed_meta: cv2.dilate(trimmed_meta, kernel_dilation))
    df_img['x_centroids'], df_img['y_centroids'], df_img['is_dot'] = zip(
        *df_img.apply(lambda row: extract_centroids_and_determine_type(row['dilated_metadata'], row['file_name']), 1))
    df_loss_centroids_extraction, loss_centroids_extraction = record_loss(df_img,
                                                                          'metadata_translation.determine_leftside_metadata_grid_mapping.extract_centroids_and_determine_type',
                                                                          subdir_location)

    # Remove files whose centroid metadata extraction was not successful
    df_img = df_img[~loss_centroids_extraction]

    # Determine metadata mapping for dot-type metadata and num-type metadata
    df_dot_subset = df_img[np.array(df_img['is_dot'])]
    df_num_subset = df_img[np.invert(np.array(df_img['is_dot']))]

    list_x_dot, list_y_dot, list_x_digit, list_y_digit = [0], [0], [0], [0]
    if not df_dot_subset.empty:
        list_x_dot = list(chain(*df_dot_subset['x_centroids'].tolist()))
        list_y_dot = list(chain(*df_dot_subset['y_centroids'].tolist()))

    if not df_num_subset.empty:
        list_x_digit = list(chain(*df_num_subset['x_centroids'].tolist()))
        list_y_digit = list(chain(*df_num_subset['y_centroids'].tolist()))
    dict_mapping, dict_hist = get_leftside_metadata_grid_mapping(list_x_dot, list_y_dot, list_x_digit, list_y_digit,
                                                                 subdir_location) #from metadata_translation.leftside_metadata_grid_mapping

    # Determine the value of metadata based on the mappings
    df_img['dict_metadata'] = 'empty'
    df_img['dict_metadata'] = df_img.apply(lambda row:
                                           map_coord_to_metadata(row['x_centroids'], row['y_centroids'],
                                                                 dict_mapping['dict_cat_dot'],
                                                                 dict_mapping['dict_num_dot']) if row['is_dot']
                                           else map_coord_to_metadata(row['x_centroids'], row['y_centroids'],
                                                                      dict_mapping['dict_cat_digit'],
                                                                      dict_mapping['dict_num_digit']), 1)
    df_loss_mapping, loss_mapping = record_loss(df_img,
                                                'metadata_translation.translate_leftside_metadata.map_coord_to_metadata',
                                                subdir_location)
    df_img = df_img[~loss_mapping]

    df_loss = pd.concat([df_loss_centroids_extraction, df_loss_mapping])

    return df_img, df_loss, dict_mapping, dict_hist