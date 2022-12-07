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
import glob
import ntpath
import os

import numpy as np
import pandas as pd

from image_segmentation.segment_images_in_subdir import segment_images
from ionogram_content_extraction.extract_all_coordinates_ionogram_trace import extract_coord_subdir_and_param
from ionogram_grid_determination.grid_mapping import all_stack, get_grid_mappings
from metadata_translation.translate_leftside_metadata import get_leftside_metadata, get_bottomside_metadata


# from helper_functions import record_loss, generate_random_subdirectory, generate_random_image_from_subdirectory


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
        df_img_bottom, df_loss_meta_bottom, dict_mapping_bottom, dict_hist_bottom = get_bottomside_metadata(df_img_bottom, subdir_path) #from metadata_translation.translate_leftside_metadata
        df_loss = df_loss.append(df_loss_meta_bottom)
        #Extract the coordinates of the ionogram trace (black), Map the (x,y) pixel coordinates to (Hz, km) values
        df_processed_bottom, df_loss_coord_bottom = extract_coord_subdir_and_param(df_img_bottom, subdir_path, col_peaks, row_peaks, mapping_Hz, mapping_km) #from ionogram_content_extraction.extract_all_coordinates_ionogram_trace
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


'''def process_subdirectory_OLD(subdir_path,
                         regex_images,
                         output_folder_if_pickle,
                         min_n_leftside_metadata=10,
                         only_ionogram_content_extraction_on_leftside_metadata=True,
                         to_pickle=True):
    """Transform raw scanned images in a subdirectory into information

    :param subdir_path: path of subdir_path
    :type subdir_path: str
    :param regex_img: regular expression to extract images ex: '*.png'
    :type regex_img: str
    :param output_folder_if_pickle: output folder for pickle, use None if no pickle
    :type output_folder_if_pickle: string
    :param min_n_leftside_metadata: minimum number of ionograms with metadata on the left to be able to call metadata_translation.leftside_metadata_grid_mapping, defaults to 10
    :type min_n_leftside_metadata: int, optional
    :param only_ionogram_content_extraction_on_leftside_metadata: only run the scripts of ionogran_content_extraction on ionograms with metadata on the left, defaults to True
    :type only_ionogram_content_extraction_on_leftside_metadata: boolean, optional
    :param to_pickle: whether to save result of subdirectory processing as a pickle file, defaults to True
    :type to_pickle: boolean, optional
    :returns: df_processed, df_all_loss, df_outlier: :  dataframe containing data from running the full processing pipeline,dataframe containing file names leading to runtime errors, dataframe containing file names that do not pass pre-established filters (metadata size, ionogram size)
    :rtype: class: `pandas.core.frame.DataFrame`,class: `pandas.core.frame.DataFrame`,class: `pandas.core.frame.DataFrame`
    """
    # Run segment_images on the subdirectory
    df_img, df_loss, df_outlier = segment_images(subdir_path, regex_images) #from segment_images_in_subdir.py

    # Determine ionogram grid mappings used to map (x,y) pixel coordinates of ionogram trace to (Hz, km) values
    stack = all_stack(df_img) #from grid_mapping.py
    # Roksana comment: the logic should be reviewed
    # if stack would be empty the below function faced with error. It happens when the directory has only one file.
    col_peaks, row_peaks, mapping_Hz, mapping_km = get_grid_mappings(stack) #from grid_mapping

    # Translate metadata located on the left
    df_img_left = df_img.loc[df_img['metadata_type'] == 'left']
    
    #If there is any leftside metadata, then get_leftside_metadata, otherwise get_bottomside_metadata
    ##BUT, why not split df_img into df_img_left and df_img_bottom, and process them separately, then recombine? ****
    if len(df_img_left) > 0:  # min_n_leftside_metadata:
        # Determine leftside metadata grid (pixel coordinates to number, category mappings)
        df_img_left, df_loss_meta, dict_mapping, dict_hist = get_leftside_metadata(df_img_left, subdir_path) #from metadata_translation.translate_leftside_metadata
        df_all_loss = df_loss.append(df_loss_meta)
    else:
        df_img, df_loss_meta, dict_mapping, dict_hist = get_bottomside_metadata(df_img, subdir_path) #from metadata_translation.translate_leftside_metadata
        df_all_loss = df_loss

    #  Extract the coordinates of the ionogram trace (black), Map the (x,y) pixel coordinates to (Hz, km) values and Extract select parameters i.e. fmin
    if only_ionogram_content_extraction_on_leftside_metadata:
        df_processed, df_loss_coord = extract_coord_subdir_and_param(df_img_left, subdir_path, col_peaks, row_peaks,
                                                                     mapping_Hz, mapping_km) #from ionogram_content_extraction.extract_all_coordinates_ionogram_trace
    else:
        df_processed, df_loss_coord = extract_coord_subdir_and_param(df_img, subdir_path, col_peaks, row_peaks,
                                                                     mapping_Hz, mapping_km) #from ionogram_content_extraction.extract_all_coordinates_ionogram_trace

    df_processed['mapping_Hz'] = [mapping_Hz] * len(df_processed.index)
    df_processed['mapping_km'] = [mapping_km] * len(df_processed.index)

    if to_pickle:
        start, subdir_name = ntpath.split(subdir_path[:-1])
        start, dir_name = ntpath.split(start)
        df_processed.to_pickle(os.pardir + '/pickle/' + str(dir_name) + '_' + str(subdir_name) + '.pkl')

    df_all_loss = df_all_loss.append(df_loss_coord)
    return df_processed, df_all_loss, df_outlier'''


def process_df_leftside_metadata(df_processed, subdir_name, source_dir, is_dot):
    """Process dataframe of subdirectory containing raw scanned images with leftside metadata

    :param df_processed: regular expression to extract images ex: '*.png'
    :type df_processed: class: `pandas.core.frame.DataFrame`
    :param subdir_name: name of subdirectory
    :type subdir_path: str
    :param is_dot: whether the dataframe contains metadata
    :type is_dot: bool

    :returns: df_final_data :  dataframe containing data
    :rtype: class: `pandas.core.frame.DataFrame`
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
        df_final_data['day'] = df_final_data['day_1'] + df_final_data['day_2'] + df_final_data['day_3']
        df_final_data['hour'] = df_final_data['hour_1'] + df_final_data['hour_2']
        df_final_data['minute'] = df_final_data['minute_1'] + df_final_data['minute_2']
        df_final_data['second'] = df_final_data['second_1'] + df_final_data['second_2']
        df_final_data['station_number'] = df_final_data['station_code']
        df_final_data.drop(
            columns=['day_1', 'day_2', 'day_3', 'hour_1', 'hour_2', 'minute_1', 'minute_2', 'second_1', 'second_2'],
            axis=1, inplace=True)
        code_list_of_station = pd.read_csv(source_dir + 'Pre_1963_Code_List_Station.csv')
        df_final_result = pd.merge(df_final_data, code_list_of_station, on='station_number')
    else:
        df_final_data['year'] = df_final_data['year'] + 1960
        df_final_data['day'] = df_final_data['day_1'] + df_final_data['day_2'] + df_final_data['day_3']
        df_final_data['hour'] = df_final_data['hour_1'] + df_final_data['hour_2']
        df_final_data['minute'] = df_final_data['minute_1'] + df_final_data['minute_2']
        df_final_data['second'] = df_final_data['second_1'] + df_final_data['second_2']
        df_final_data['station_number'] = df_final_data['station_number_1'] + df_final_data['station_number_2']
        df_final_data.drop(columns=['station_number_1', 'station_number_2'], axis=1, inplace=True)
        df_final_data.drop(
            columns=['day_1', 'day_2', 'day_3', 'hour_1', 'hour_2', 'minute_1', 'minute_2', 'second_1', 'second_2'],
            axis=1, inplace=True)
        # if len(df_final_data['year']) != 0:  # and df_final_data['year'] >= 1965:
        #     code_list_of_station = pd.read_csv(source_dir+'Post_July_1_1965_Code_List_Station.csv')
        # else:
        #     code_list_of_station = pd.read_csv(source_dir + 'Pre_July_1_1965_Code_List_Station.csv')
        #
        # df_final_result = pd.merge(df_final_data,code_list_of_station, on='station_number')
        code_list_of_station_after1965 = pd.read_csv(source_dir + 'Post_July_1_1965_Code_List_Station.csv')
        code_list_of_station_before1963 = pd.read_csv(source_dir + 'Pre_1963_Code_List_Station.csv')
        code_list_of_station_between1963_1964 = pd.read_csv(source_dir + '1963_1964.csv')
        df_result_after1965 = pd.merge(df_final_data.loc[df_final_data['year'] >= 1965], code_list_of_station_after1965,
                                       on='station_number')
        df_result_before1963 = pd.merge(df_final_data.loc[df_final_data['year'] <= 1963],
                                        code_list_of_station_before1963, on='station_number')
        df_result_mid1964 = pd.merge (df_final_data.loc[df_final_data['year'] == 1964],
                                        code_list_of_station_between1963_1964, on = 'station_number')
        df_final_result = df_result_before1963.append(df_result_after1965.append(df_result_mid1964, ignore_index=True))

    return df_final_result


def process_df_bottomside_metadata(df_processed, subdir_name, source_dir):
    """Process dataframe of subdirectory containing raw scanned images with leftside metadata

    :param df_processed: regular expression to extract images ex: '*.png'
    :type df_processed: class: `pandas.core.frame.DataFrame`
    :param subdir_name: name of subdirectory
    :type subdir_path: str

    :returns: df_final_data :  dataframe containing data
    :rtype: class: `pandas.core.frame.DataFrame`
    """

    df_final_data = df_processed[['file_name', 'fmin', 'max_depth', 'dict_metadata', 'mapped_coord']]
    df_final_data['subdir_name'] = subdir_name
    labels = ['satellite_number', 'year', 'day_1', 'day_2', 'day_3', 'hour_1', 'hour_2', 'minute_1', 'minute_2',
              'second_1', 'second_2', 'station_number_1', 'station_number_2']

    for label in labels:
        df_final_data[label] = df_final_data['dict_metadata'].map(
            lambda dict_meta: sum(dict_meta[label]) if label in dict_meta.keys() else 0)

    del df_final_data['dict_metadata']

    df_final_data['year'] = df_final_data['year'] + 1960
    df_final_data['day'] = df_final_data['day_1'] + df_final_data['day_2'] + df_final_data['day_3']
    df_final_data['hour'] = df_final_data['hour_1'] + df_final_data['hour_2']
    df_final_data['minute'] = df_final_data['minute_1'] + df_final_data['minute_2']
    df_final_data['second'] = df_final_data['second_1'] + df_final_data['second_2']
    df_final_data['station_number'] = df_final_data['station_number_1'] + df_final_data['station_number_2']
    df_final_data.drop(columns=['station_number_1', 'station_number_2'], axis=1, inplace=True)
    df_final_data.drop(
        columns=['day_1', 'day_2', 'day_3', 'hour_1', 'hour_2', 'minute_1', 'minute_2', 'second_1', 'second_2'],
        axis=1, inplace=True)
    # if len(df_final_data['year']) != 0:  # and df_final_data['year'] >= 1965:
    #     code_list_of_station = pd.read_csv(source_dir+'Post_July_1_1965_Code_List_Station.csv')
    # else:
    #     code_list_of_station = pd.read_csv(source_dir + 'Pre_July_1_1965_Code_List_Station.csv')
    #
    # df_final_result = pd.merge(df_final_data,code_list_of_station, on='station_number')
    code_list_of_station_after1965 = pd.read_csv(source_dir + 'Post_July_1_1965_Code_List_Station.csv')
    code_list_of_station_before1963 = pd.read_csv(source_dir + 'Pre_1963_Code_List_Station.csv')
    code_list_of_station_between1963_1964 = pd.read_csv(source_dir + '1963_1964.csv')
    df_result_after1965 = pd.merge(df_final_data.loc[df_final_data['year'] >= 1965], code_list_of_station_after1965,
                                   on='station_number')
    df_result_before1963 = pd.merge(df_final_data.loc[df_final_data['year'] <= 1963],
                                    code_list_of_station_before1963, on='station_number')
    df_result_mid1964 = pd.merge(df_final_data.loc[df_final_data['year'] == 1964],
                                 code_list_of_station_between1963_1964, on='station_number')
    df_final_result = df_result_before1963.append(df_result_after1965.append(df_result_mid1964, ignore_index=True))

    return df_final_result


'''def append_to_csv(csv_path, df):
    #Append select data from a dataframe to a master CSV
    
    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_path, mode='a', index=False)'''


'''if __name__ == '__main__':
    #    subdir_path = generate_random_subdirectory(regex_subdirectory='E:/master/R*/[0-9]*/')

    list_all_subdir = glob.glob('C:/Users/Hansen/Desktop/Projects/CSA Internship/AlouetteIonogramsRawData/R*/[0-9]*/')
    regex_raw = '*.png'

    # list_all_subdir2 = glob.glob('G:/AlouetteData/Alouette Data/R*/[0-9]*[0-9]/')
    # regex_raw2= 'Image*[0-9].png'

    dir_csv_output = 'C:/Users/Hansen/Desktop/'

    for subdir_path in list_all_subdir:
        processed, all_loss, outlier = process_subdirectory(subdir_path, regex_raw, dir_csv_output,
                                                            only_ionogram_content_extraction_on_leftside_metadata=True,
                                                            to_pickle=True)

        list_metadata_type = processed['metadata_type'].tolist()
        type_metadata_subdir = max(set(list_metadata_type), key=list_metadata_type.count)

        if type_metadata_subdir == 'left':
            is_dot = np.array(processed['is_dot'])
            df_dot_subset = processed[is_dot]
            df_num_subset = processed[np.invert(is_dot)]
            start, subdir_name = ntpath.split(subdir_path[:-1])
            df_dot_subset = process_df_leftside_metadata(df_dot_subset, subdir_name, is_dot=True)
            df_num_subset = process_df_leftside_metadata(df_num_subset, subdir_name, is_dot=False)
            append_to_csv(dir_csv_output + 'dot_data.csv', df_dot_subset)
            append_to_csv(dir_csv_output + 'num_data.csv', df_num_subset)

        append_to_csv(dir_csv_output + 'loss.csv', all_loss)
        append_to_csv(dir_csv_output + 'outlier.csv', outlier)'''


#Remove only leftside processing
def process_extract_management(dir_csv_output, master_dir, regex_raw, sample_subdir):
    
    df_processed, df_loss, df_outlier = process_subdirectory(sample_subdir, regex_raw)

    # Split left from bottom-side metadata
    df_processed_left = df_processed.loc[df_processed['metadata_type'] == 'left']
    df_processed_bottom = df_processed.loc[df_processed['metadata_type'] == 'bottom']

    df_dot = pd.DataFrame()
    df_num = pd.DataFrame()
    if len(df_processed_left) > 0:
        is_dot = np.array(df_processed_left['is_dot'])
        df_dot_subset = df_processed_left[is_dot]
        df_num_subset = df_processed_left[np.invert(is_dot)]
        start, subdir_name = ntpath.split(sample_subdir[:-1])
        df_dot_subset = process_df_leftside_metadata(df_dot_subset, subdir_name, master_dir, is_dot=True)
        df_num_subset = process_df_leftside_metadata(df_num_subset, subdir_name, master_dir, is_dot=False)
        df_dot = pd.concat([df_dot, df_dot_subset])
        df_num = pd.concat([df_num, df_num_subset])
    
    #***Is there bottom-side dot metadata?***
    if len(df_processed_bottom) > 0:
        is_dot = np.array(df_processed_bottom['is_dot'])
        df_dot_subset = df_processed_bottom[is_dot]
        df_num_subset = df_processed_bottom[np.invert(is_dot)]
        start, subdir_name = ntpath.split(sample_subdir[:-1])
        df_loss = pd.concat([df_loss, df_dot_subset]) #df_dot_subset = process_df_bottomside_metadata(df_dot_subset, subdir_name, master_dir, is_dot=True)
        df_num_subset = process_df_bottomside_metadata(df_num_subset, subdir_name, master_dir)
        #df_dot = pd.concat([df_dot, df_dot_subset])
        df_num = pd.concat([df_num, df_num_subset])

    #Save dataframes
    df_dot.to_csv(dir_csv_output + 'df_dot.csv', index=False)
    df_num.to_csv(dir_csv_output + 'df_num.csv', index=False)
    df_loss.to_csv(dir_csv_output + 'df_loss.csv', index=False)
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
        
        
        

'''def process_extract_management_OLD(dir_csv_output, master_dir, regex_raw, sample_subdir, desire_pickle=False,
                               desire_only_leftside=True):
    if desire_pickle:
        to_pickle_input = True
    else:
        to_pickle_input = False
    if desire_only_leftside:
        only_leftside_input = True
    else:
        only_leftside_input = False
    processed, all_loss, outlier = process_subdirectory(sample_subdir,
                                                        regex_raw,
                                                        dir_csv_output,
                                                        only_ionogram_content_extraction_on_leftside_metadata=only_leftside_input,
                                                        to_pickle=to_pickle_input)
    #processed.to_csv(dir_csv_output + 'df_processed.csv')

    #list_metadata_type = processed['metadata_type'].tolist()
    #type_metadata_subdir = max(set(list_metadata_type), key=list_metadata_type.count)
    
    if type_metadata_subdir == 'left':
        is_dot = np.array(processed['is_dot'])
        df_dot_subset = processed[is_dot]
        df_num_subset = processed[np.invert(is_dot)]  # not sure if this works
        start, subdir_name = ntpath.split(sample_subdir[:-1])
        df_dot_subset = process_df_leftside_metadata(df_dot_subset, subdir_name, master_dir, is_dot=True)
        df_num_subset = process_df_leftside_metadata(df_num_subset, subdir_name, master_dir, is_dot=False)
        append_to_csv(dir_csv_output + 'dot_data.csv', df_dot_subset)
        append_to_csv(dir_csv_output + 'num_data.csv', df_num_subset)
    else:
        start, subdir_name = ntpath.split(sample_subdir[:-1])
        df_num_subset = process_df_bottomside_metadata(processed, subdir_name, master_dir)
        append_to_csv(dir_csv_output + 'num_data.csv', df_num_subset)
    append_to_csv(dir_csv_output + 'loss.csv', all_loss)
    append_to_csv(dir_csv_output + 'outlier.csv', outlier)
    
    #Save mapped coordinates
    for i in range(0, len(processed)):
        path = processed['file_name'].iloc[i].replace(master_dir, '')
        path = path.replace('/', '\\')
        path = path.replace('\\', '/')
        parts = path.split('/')
        parts[2] = parts[2].replace('.png', '')
        newDir = dir_csv_output + parts[0] + '/' + parts[1] + '/'
        os.makedirs(newDir, exist_ok=True)
        np.save(newDir + 'mapped_coords-' + parts[0] + '_' + parts[1] + '_' + parts[2], processed['mapped_coord'].iloc[i])'''
