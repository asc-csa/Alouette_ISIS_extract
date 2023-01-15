# -*- coding: utf-8 -*-
"""

"""
import ntpath
import pandas as pd

from image_segmentation.segment_images_in_subdir import segment_images
#from ionogram_content_extraction.extract_all_coordinates_ionogram_trace import extract_coord_subdir_and_param
#from ionogram_grid_determination.grid_mapping import all_stack, get_grid_mappings
from metadata_translation.translate_leftside_metadata import get_leftside_metadata, get_bottomside_metadata



def process_subdirectory_md(subdir_path, regex_images):

    # Run segment_images on the subdirectory
    df_img, df_loss, df_outlier = segment_images(subdir_path, regex_images) #from image_segmentation.segment_images_in_subdir.py

    # Determine ionogram grid mappings used to map (x,y) pixel coordinates of ionogram trace to (Hz, km) values
    #stack = all_stack(df_img) #from grid_mapping.py
    # if stack would be empty the below function faced with error. It happens when the directory has only one file.
    #col_peaks, row_peaks, mapping_Hz, mapping_km = get_grid_mappings(stack) #from grid_mapping

    # Split left from bottom-side metadata
    df_img_left = df_img.loc[df_img['metadata_type'] == 'left']
    df_img_bottom = df_img.loc[df_img['metadata_type'] == 'bottom']
    
    if len(df_img_left) > 9:
        #Get metadata
        df_img_left, df_loss_meta_left, dict_mapping_left, dict_hist_left = get_leftside_metadata(df_img_left, subdir_path) #from metadata_translation.translate_leftside_metadata
        df_loss = df_loss.append(df_loss_meta_left)
        #Extract the coordinates of the ionogram trace (black), Map the (x,y) pixel coordinates to (Hz, km) values
        #df_processed_left, df_loss_coord_left = extract_coord_subdir_and_param(df_img_left, subdir_path, col_peaks, row_peaks, mapping_Hz, mapping_km) #from ionogram_content_extraction.extract_all_coordinates_ionogram_trace
    else:
        df_loss = df_loss.append(df_img_left)
        #df_processed_left = pd.DataFrame()
        #df_loss_coord_left = pd.DataFrame()
    
    if len(df_img_bottom) > 9:
        df_img_bottom, df_loss_meta_bottom, dict_mapping_bottom, dict_hist_bottom = get_bottomside_metadata(df_img_bottom, subdir_path) #from metadata_translation.translate_leftside_metadata
        df_loss = df_loss.append(df_loss_meta_bottom)
        #Extract the coordinates of the ionogram trace (black), Map the (x,y) pixel coordinates to (Hz, km) values
        #df_processed_bottom, df_loss_coord_bottom = extract_coord_subdir_and_param(df_img_bottom, subdir_path, col_peaks, row_peaks, mapping_Hz, mapping_km) #from ionogram_content_extraction.extract_all_coordinates_ionogram_trace
    else:
        df_loss = df_loss.append(df_img_bottom)
        #df_processed_bottom = pd.DataFrame()
        #df_loss_coord_bottom = pd.DataFrame()

    #Recombine left and bottom-side metadata images
    df_processed = pd.concat([df_img_left, df_img_bottom])
    #df_processed = pd.concat([df_processed_left, df_processed_bottom])
    #df_loss_coord = pd.concat([df_loss_coord_left, df_loss_coord_bottom])
    
    #df_processed['mapping_Hz'] = [mapping_Hz] * len(df_processed.index)
    #df_processed['mapping_km'] = [mapping_km] * len(df_processed.index)

    #df_loss = df_loss.append(df_loss_coord)
    return df_processed, df_loss, df_outlier



def process_df_leftside_metadata(df_processed, subdir_name, source_dir, is_dot):

    df_final_data = df_processed[['file_name', 'dict_metadata']]
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
        df_final_data['day'] = df_final_data['day_1'].astype(str) + df_final_data['day_2'].astype(str) + df_final_data['day_3'].astype(str) 
        df_final_data['hour'] = df_final_data['hour_1'].astype(str) + df_final_data['hour_2'].astype(str) 
        df_final_data['minute'] = df_final_data['minute_1'].astype(str) + df_final_data['minute_2'].astype(str) 
        df_final_data['second'] = df_final_data['second_1'].astype(str) + df_final_data['second_2'].astype(str) 
        df_final_data['station_number'] = df_final_data['station_code']
        df_final_data['day'] = df_final_data['day'].astype(int)
        df_final_data['hour'] = df_final_data['hour'].astype(int)
        df_final_data['minute'] = df_final_data['minute'].astype(int)
        df_final_data['second'] = df_final_data['second'].astype(int)
        df_final_data['station_number'] = df_final_data['station_number'].astype(int)
        code_list_of_station = pd.read_csv(source_dir + 'Pre_1963_Code_List_Station.csv')
        df_final_result = pd.merge(df_final_data, code_list_of_station, on='station_number')
    else:
        df_final_data['year'] = df_final_data['year'] + 1960
        df_final_data['day'] = df_final_data['day_1'].astype(str) + df_final_data['day_2'].astype(str) + df_final_data['day_3'].astype(str) 
        df_final_data['hour'] = df_final_data['hour_1'].astype(str) + df_final_data['hour_2'].astype(str)
        df_final_data['minute'] = df_final_data['minute_1'].astype(str) + df_final_data['minute_2'].astype(str)
        df_final_data['second'] = df_final_data['second_1'].astype(str) + df_final_data['second_2'].astype(str)
        df_final_data['station_number'] = df_final_data['station_number_1'].astype(str) + df_final_data['station_number_2'].astype(str)
        df_final_data['day'] = df_final_data['day'].astype(int)
        df_final_data['hour'] = df_final_data['hour'].astype(int)
        df_final_data['minute'] = df_final_data['minute'].astype(int)
        df_final_data['second'] = df_final_data['second'].astype(int)
        df_final_data['station_number'] = df_final_data['station_number'].astype(int)

        if len(df_final_data) > 0:          
            code_list_of_station_after1965 = pd.read_csv(source_dir + 'Post_July_1_1965_Code_List_Station.csv')
            code_list_of_station_before1963 = pd.read_csv(source_dir + 'Pre_1963_Code_List_Station.csv')
            code_list_of_station_between1963_1964 = pd.read_csv(source_dir + '1963_1964.csv')
            df_result_after1965 = pd.merge(df_final_data.loc[df_final_data['year'] >= 1965], 
                                           code_list_of_station_after1965, on='station_number')
            df_result_before1963 = pd.merge(df_final_data.loc[df_final_data['year'] <= 1963],
                                            code_list_of_station_before1963, on='station_number')
            df_result_mid1964 = pd.merge (df_final_data.loc[df_final_data['year'] == 1964],
                                            code_list_of_station_between1963_1964, on = 'station_number')
            df_final_result = df_result_before1963.append(df_result_after1965.append(df_result_mid1964, ignore_index=True)) #Why was pd.concat not used?

    return df_final_result



def process_df_bottomside_metadata(df_processed, subdir_name, source_dir):

    df_final_data = df_processed[['file_name', 'dict_metadata']]
    df_final_data['subdir_name'] = subdir_name
    labels = ['satellite_number', 'year', 'day_1', 'day_2', 'day_3', 'hour_1', 'hour_2', 'minute_1', 'minute_2',
              'second_1', 'second_2', 'station_number_1', 'station_number_2']

    for label in labels:
        df_final_data[label] = df_final_data['dict_metadata'].map(
            lambda dict_meta: sum(dict_meta[label]) if label in dict_meta.keys() else 0)

    del df_final_data['dict_metadata']

    df_final_data['year'] = df_final_data['year'] + 1960
    df_final_data['day'] = df_final_data['day_1'].astype(str) + df_final_data['day_2'].astype(str) + df_final_data['day_3'].astype(str) 
    df_final_data['hour'] = df_final_data['hour_1'].astype(str) + df_final_data['hour_2'].astype(str) 
    df_final_data['minute'] = df_final_data['minute_1'].astype(str) + df_final_data['minute_2'].astype(str) 
    df_final_data['second'] = df_final_data['second_1'].astype(str) + df_final_data['second_2'].astype(str) 
    df_final_data['station_number'] = df_final_data['station_number_1'].astype(str) + df_final_data['station_number_2'].astype(str) 
    df_final_data['day'] = df_final_data['day'].astype(int)
    df_final_data['hour'] = df_final_data['hour'].astype(int)
    df_final_data['minute'] = df_final_data['minute'].astype(int)
    df_final_data['second'] = df_final_data['second'].astype(int)
    df_final_data['station_number'] = df_final_data['station_number'].astype(int)

    if len(df_final_data) > 0:          
        code_list_of_station_after1965 = pd.read_csv(source_dir + 'Post_July_1_1965_Code_List_Station.csv')
        code_list_of_station_before1963 = pd.read_csv(source_dir + 'Pre_1963_Code_List_Station.csv')
        code_list_of_station_between1963_1964 = pd.read_csv(source_dir + '1963_1964.csv')
        df_result_after1965 = pd.merge(df_final_data.loc[df_final_data['year'] >= 1965], 
                                       code_list_of_station_after1965, on='station_number')
        df_result_before1963 = pd.merge(df_final_data.loc[df_final_data['year'] <= 1963],
                                        code_list_of_station_before1963, on='station_number')
        df_result_mid1964 = pd.merge (df_final_data.loc[df_final_data['year'] == 1964],
                                        code_list_of_station_between1963_1964, on = 'station_number')
        df_final_result = df_result_before1963.append(df_result_after1965.append(df_result_mid1964, ignore_index=True)) #Why was pd.concat not used?        
    
    return df_final_result



def process_extract_management(dir_csv_output, master_dir, regex_raw, sample_subdir):
    
    df_processed, df_loss, df_outlier = process_subdirectory_md(sample_subdir, regex_raw)
    df_processed = df_processed.dropna(subset=['dict_metadata'])
    
    # Split left from bottom-side metadata
    df_processed_left = df_processed.loc[df_processed['metadata_type'] == 'left']
    df_processed_bottom = df_processed.loc[df_processed['metadata_type'] == 'bottom']

    df_dot = pd.DataFrame()
    df_num = pd.DataFrame()
    if len(df_processed_left) > 0: 
        df_dot_subset = df_processed_left.loc[df_processed_left['is_dot'] == True] 
        df_num_subset = df_processed_left.loc[df_processed_left['is_dot'] == False]
        start, subdir_name = ntpath.split(sample_subdir[:-1])
        df_dot_subset = process_df_leftside_metadata(df_dot_subset, subdir_name, master_dir, is_dot=True)
        df_num_subset = process_df_leftside_metadata(df_num_subset, subdir_name, master_dir, is_dot=False)
        df_dot = pd.concat([df_dot, df_dot_subset])
        df_num = pd.concat([df_num, df_num_subset])
    
    #***Is there bottom-side dot metadata?***
    if len(df_processed_bottom) > 0:
        df_dot_subset = df_processed_bottom.loc[df_processed_bottom['is_dot'] == True] 
        df_num_subset = df_processed_bottom.loc[df_processed_bottom['is_dot'] == False] 
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
    
