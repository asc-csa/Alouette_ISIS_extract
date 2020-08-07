# -*- coding: utf-8 -*-
"""
Interact with user input through command line

"""
import glob
import ntpath
import numpy as np
import process_directory


def main():
    dir_csv_output =  'C:/Users/WMohammad/Desktop/' #input('Type directory for outputs of processing ex: C:/Users/JPatel/Desktop/: ')
    master_dir = 'D:/master/leftside_data/' #input('Type directory with all the raw data ex: E:/master/: ')
    regex_raw = '*.png'

    desire = input('Enter Y if you you want to process all the dataset: ')
    if desire == 'Y':
        list_all_subdir = glob.glob(master_dir+'R*/[0-9]*/')
        for sample_subdir in list_all_subdir:
            desire_pickle = 'N' #input('Enter Y if you want to save the processed data as pickle files: ')
            if desire_pickle == 'Y':
                to_pickle_input= True
            else:
                to_pickle_input= False

            desire_only_leftside = 'Y' #input('Enter Y if you only want to process ionograms with metadata on the left: ')
            if desire_only_leftside == 'Y':
                only_leftside_input= True
            else:
                only_leftside_input= False
            processed, all_loss, outlier = process_directory.process_subdirectory(sample_subdir, regex_raw, dir_csv_output,
                                            only_ionogram_content_extraction_on_leftside_metadata=only_leftside_input, to_pickle=to_pickle_input)

            list_metadata_type = processed['metadata_type'].tolist()
            type_metadata_subdir = max(set(list_metadata_type), key=list_metadata_type.count)

            if type_metadata_subdir == 'left':
                is_dot = np.array(processed['is_dot'])
                df_dot_subset = processed[is_dot]
                df_num_subset = processed[np.invert(is_dot)]
                start,subdir_name = ntpath.split(sample_subdir[:-1])
                df_dot_subset = process_directory.process_df_leftside_metadata(df_dot_subset,subdir_name,is_dot=True)
                df_num_subset = process_directory.process_df_leftside_metadata(df_num_subset,subdir_name,is_dot=False)
                process_directory.append_to_csv(dir_csv_output+'dot_data.csv', df_dot_subset)
                process_directory.append_to_csv(dir_csv_output+'num_data.csv', df_num_subset)

            process_directory.append_to_csv(dir_csv_output+'loss.csv', all_loss)
            process_directory.append_to_csv(dir_csv_output+'outlier.csv', outlier)
    else:
        sample_subdir = 'D:/master/leftside_data/R014207938/907-A/' #input('Enter the name of the subdirectory ex: E:/master/R014207968/1246-5A/: ')
        desire_pickle = 'N' #input('Enter Y if you want to save the processed data as pickle files: ')
        if desire_pickle == 'Y':
            to_pickle_input= True
        else:
            to_pickle_input= False

        desire_only_leftside = 'Y' #input('Enter Y if you only want to process ionograms with metadata on the left: ')
        if desire_only_leftside == 'Y':
            only_leftside_input= True
        else:
            only_leftside_input= False
        processed, all_loss, outlier = process_directory.process_subdirectory(sample_subdir, regex_raw, dir_csv_output,
                                                                              only_ionogram_content_extraction_on_leftside_metadata=only_leftside_input,
                                                                              to_pickle=to_pickle_input)
        list_metadata_type = processed['metadata_type'].tolist()
        type_metadata_subdir = max(set(list_metadata_type), key=list_metadata_type.count)

        if type_metadata_subdir == 'left':
            is_dot = np.array(processed['is_dot'])
            df_dot_subset = processed[is_dot]
            df_num_subset = processed[np.invert(is_dot)] #not sure if this works
            start,subdir_name = ntpath.split(sample_subdir[:-1])
            df_dot_subset = process_directory.process_df_leftside_metadata(df_dot_subset,subdir_name,is_dot=True)
            df_num_subset = process_directory.process_df_leftside_metadata(df_num_subset,subdir_name,is_dot=False)
            process_directory.append_to_csv(dir_csv_output+'dot_data.csv', df_dot_subset)
            process_directory.append_to_csv(dir_csv_output+'num_data.csv', df_num_subset)

        process_directory.append_to_csv(dir_csv_output+'loss.csv', all_loss)
        process_directory.append_to_csv(dir_csv_output+'outlier.csv', outlier)


if __name__ == "__main__":
    main()