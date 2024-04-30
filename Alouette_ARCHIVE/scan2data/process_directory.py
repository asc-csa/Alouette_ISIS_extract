# -*- coding: utf-8 -*-
"""
Process all the raw images in a subdirectory
- Determine ionogram grid (pixel coordinates to Hz, km mappings)
- Determine leftside metadata grid (pixel coordinates to number, category mappings)
- For each raw image in the subdirectory,
    - Segment the raw image into raw ionogram and raw metadata
    - Trim the metadata
    - Translate the leftside metadata into information
    - Extract the coordinates of the ionogram trace (black)
    - Map the (x,y) pixel coordinates to (Hz, km) values
    - Extract select parameters i.e. fmin
"""
import ntpath
import os

import numpy as np
import pandas as pd

from image_segmentation.segment_images_in_subdir import segment_images
from ionogram_content_extraction.extract_all_coordinates_ionogram_trace import extract_coord_subdir_and_param
from ionogram_grid_determination.grid_mapping import all_stack, get_grid_mappings
from metadata_translation.translate_leftside_metadata import get_leftside_metadata, get_bottomside_metadata



def process_subdirectory(subdir_path, regex_images):
    """Transform raw scanned images in a subdirectory into information

    :param subdir_path: path of subdir_path
    :type subdir_path: str
    :param regex_img: regular expression to extract images ex: '*.png'
    :type regex_img: str
    :param output_folder_if_pickle: output folder for pickle, use None if no pickle
    :type output_folder_if_pickle: string
    :param min_n_leftside_metadata: minimum number of ionograms with metadata on the left to be able to call metadata_translation.leftside_metadata_grid_mapping, defaults to 10
    :type min_n_leftside_metadata: int, optional
    :returns: df_processed, df_loss, df_outlier: :  dataframe containing data from running the full processing pipeline,dataframe containing file names leading to runtime errors, dataframe containing file names that do not pass pre-established filters (metadata size, ionogram size)
    :rtype: class: `pandas.core.frame.DataFrame`,class: `pandas.core.frame.DataFrame`,class: `pandas.core.frame.DataFrame`
    """
    # Run segment_images on the subdirectory
    df_img, df_loss, df_outlier = segment_images(subdir_path, regex_images) #from image_segmentation.segment_images_in_subdir.py

    # Determine ionogram grid mappings used to map (x,y) pixel coordinates of ionogram trace to (Hz, km) values
    stack = all_stack(df_img) #from grid_mapping.py
    # Roksana comment: the logic should be reviewed
    # if stack would be empty the below function faced with error. It happens when the directory has only one file.
    col_peaks, row_peaks, mapping_Hz, mapping_km = get_grid_mappings(stack) #from grid_mapping

    # Split left from bottom-side metadata
    df_img_left = df_img.loc[df_img['metadata_type'] == 'left']
    df_img_bottom = df_img.loc[df_img['metadata_type'] == 'bottom']
    
    if len(df_img_left) > 9:
        #Get metadata
        df_img_left, df_loss_meta_left, dict_mapping_left, dict_hist_left = get_leftside_metadata(df_img_left, subdir_path) #from metadata_translation.translate_leftside_metadata
        df_loss = df_loss.append(df_loss_meta_left)
        #Extract the coordinates of the ionogram trace (black), Map the (x,y) pixel coordinates to (Hz, km) values
        df_processed_left, df_loss_coord_left = extract_coord_subdir_and_param(df_img_left, subdir_path, col_peaks, row_peaks, mapping_Hz, mapping_km) #from ionogram_content_extraction.extract_all_coordinates_ionogram_trace
    else:
        df_loss = df_loss.append(df_img_left)
        df_processed_left = pd.DataFrame()
        df_loss_coord_left = pd.DataFrame()
    
    if len(df_img_bottom) > 9:
        df_img_bottom, df_loss_meta_bottom, dict_mapping_bottom, dict_hist_bottom = get_bottomside_metadata(df_img_bottom, subdir_path) #from metadata_translation.translate_bottomside_metadata
        df_loss = df_loss.append(df_loss_meta_bottom)
        if len(df_img_bottom) > 0:
            #Extract the coordinates of the ionogram trace (black), Map the (x,y) pixel coordinates to (Hz, km) values
            df_processed_bottom, df_loss_coord_bottom = extract_coord_subdir_and_param(df_img_bottom, subdir_path, col_peaks, row_peaks, mapping_Hz, mapping_km) #from ionogram_content_extraction.extract_all_coordinates_ionogram_trace
        else :
            df_processed_bottom = pd.DataFrame()
            df_loss_coord_bottom = pd.DataFrame()
            
    else:
        df_loss = df_loss.append(df_img_bottom)
        df_processed_bottom = pd.DataFrame()
        df_loss_coord_bottom = pd.DataFrame()

    #Recombine left and bottom-side metadata images
    df_processed = pd.concat([df_processed_left, df_processed_bottom])
    df_loss_coord = pd.concat([df_loss_coord_left, df_loss_coord_bottom])
    
    df_processed['mapping_Hz'] = [mapping_Hz] * len(df_processed.index)
    df_processed['mapping_km'] = [mapping_km] * len(df_processed.index)

    df_loss = df_loss.append(df_loss_coord)
    return df_processed, df_loss, df_outlier



def process_df_leftside_metadata(df_processed, subdir_name, source_dir, is_dot):
    """Process dataframe of subdirectory containing raw scanned images with leftside metadata
    """

    df_final_data = df_processed[['file_name', 'fmin', 'max_depth', 'dict_metadata', 'mapped_coord']]
    df_final_data['subdir_name'] = subdir_name
    if is_dot:
        labels = ['day_1', 'day_2', 'day_3', 'hour_1', 'hour_2', 'minute_1', 'minute_2',
                  'second_1', 'second_2', 'station_code']
    else:
        labels = ['satellite_number', 'year', 'day_1', 'day_2', 'day_3', 'hour_1', 'hour_2', 'minute_1', 'minute_2',
                  'second_1', 'second_2', 'station_number_1', 'station_number_2']

    for label in labels:
        df_final_data[label] = df_final_data['dict_metadata'].map(
            lambda dict_meta: sum(dict_meta[label]) if label in dict_meta.keys() else 0)

    del df_final_data['dict_metadata']

    if is_dot:
        df_final_data['station_number'] = df_final_data['station_code']
        df_final_data['station_number'] = df_final_data['station_number'].astype(int)

    return df_final_data


def process_df_bottomside_metadata(df_processed, subdir_name, source_dir):
    """Process dataframe of subdirectory containing raw scanned images with bottomside metadata
    """

    df_final_data = df_processed[['file_name', 'fmin', 'max_depth', 'dict_metadata', 'mapped_coord']]
    df_final_data['subdir_name'] = subdir_name
    labels = ['satellite_number', 'year', 'day_1', 'day_2', 'day_3', 'hour_1', 'hour_2', 'minute_1', 'minute_2',
              'second_1', 'second_2', 'station_number_1', 'station_number_2']

    for label in labels:
        df_final_data[label] = df_final_data['dict_metadata'].map(
            lambda dict_meta: sum(dict_meta[label]) if label in dict_meta.keys() else 0)

    del df_final_data['dict_metadata']

    return df_final_data


#Remove only leftside processing
def process_extract_management(dir_csv_output, master_dir, regex_raw, sample_subdir):
    
    df_processed, df_loss, df_outlier = process_subdirectory(sample_subdir, regex_raw)

    if(len(df_processed)==0):
        df_processed_left=pd.DataFrame()
        df_processed_bottom=pd.DataFrame()
    else:
        # Split left from bottom-side metadata
        df_processed_left = df_processed.loc[df_processed['metadata_type'] == 'left']
        df_processed_bottom = df_processed.loc[df_processed['metadata_type'] == 'bottom']

    df_dot = pd.DataFrame()
    df_num = pd.DataFrame()
    
    if len(df_processed_left) > 0:
        df_dot_subset = df_processed_left.loc[df_processed_left['is_dot'] == 1.]
        df_num_subset = df_processed_left.loc[df_processed_left['is_dot'] == 0.]
        start, subdir_name = ntpath.split(sample_subdir[:-1])
        df_dot_subset = process_df_leftside_metadata(df_dot_subset, subdir_name, master_dir, is_dot=True)
        df_num_subset = process_df_leftside_metadata(df_num_subset, subdir_name, master_dir, is_dot=False)
        df_dot = pd.concat([df_dot, df_dot_subset])
        df_num = pd.concat([df_num, df_num_subset])
    
    #Assume that there is no bottom-side dot type metadata (count df_dot_subset as loss)
    if len(df_processed_bottom) > 0:
        #is_dot = np.array(df_processed_bottom['is_dot'])
        #df_dot_subset = df_processed_bottom[is_dot]
        df_dot_subset = df_processed_bottom.loc[df_processed_bottom['is_dot'] == True]
        #df_num_subset = df_processed_bottom[np.invert(is_dot)]
        df_num_subset = df_processed_bottom.loc[df_processed_bottom['is_dot'] != True]
        start, subdir_name = ntpath.split(sample_subdir[:-1])
        df_loss = pd.concat([df_loss, df_dot_subset])
        df_num_subset = process_df_bottomside_metadata(df_num_subset, subdir_name, master_dir)
        df_num = pd.concat([df_num, df_num_subset])

    #Save dataframes
    if len(df_dot) > 0:
        df_dot.to_csv(dir_csv_output + 'df_dot.csv', index=False)
    if len(df_num) > 0:
        df_num.to_csv(dir_csv_output + 'df_num.csv', index=False)
    if len(df_loss) > 0:
        df_loss.to_csv(dir_csv_output + 'df_loss.csv', index=False)
    if len(df_outlier) > 0:
        df_outlier.to_csv(dir_csv_output + 'df_outlier.csv', index=False)
    
    #Save mapped coordinates
    for i in range(0, len(df_processed)):
        path = df_processed['file_name'].iloc[i].replace(master_dir, '')
        path = path.replace('/', '\\')
        path = path.replace('\\', '/')
        parts = path.split('/')
        parts[2] = parts[2].replace('.png', '')
        newDir = dir_csv_output + 'mapped_coords/' + parts[0] + '/' + parts[1] + '/'
        os.makedirs(newDir, exist_ok=True)
        np.save(newDir + 'mapped_coords-' + parts[0] + '_' + parts[1] + '_' + parts[2], df_processed['mapped_coord'].iloc[i])
        
        