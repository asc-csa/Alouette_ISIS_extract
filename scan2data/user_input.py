# -*- coding: utf-8 -*-
"""
Interact with user input through command line

"""
import glob
import ntpath
import numpy as np
import process_directory


def main():
    processAllSubdirectories = 'Y' #will not pass to process-extract-management
    dir_csv_output ='/Users/Roksana/Desktop/Final Data/' #input('Type directory for outputs of processing ex: C:/Users/JPatel/Desktop/: ')
    master_dir = '/Users/Roksana/Desktop/Data Source/' #input('Type directory with all the raw data ex: E:/master/: ')
    regex_raw = '*.png'
    desire_pickle = 'N' #enter Y if you want to save it as a pickle file.
    desire_only_leftside = 'Y' #Enter N for it to analyze the bottom data.

    if processAllSubdirectories == 'Y':
        list_all_subdir = glob.glob(master_dir + 'R*/[0-9]*/')
        for sample_subdir in list_all_subdir:
            process_directory.process_extract_management(dir_csv_output,
                                                         master_dir,
                                                         regex_raw,
                                                         sample_subdir,
                                                         desire_pickle,
                                                         desire_only_leftside)

    else:
        sample_subdir = '/Users/Roksana/Desktop/Data Source/R014207938/907A/'  # input('Enter the name of the subdirectory ex: E:/master/R014207968/1246-5A/: ')
        process_directory.process_extract_management(dir_csv_output,
                                                     master_dir,
                                                     regex_raw,
                                                     sample_subdir,
                                                     desire_pickle,
                                                     desire_only_leftside)
if __name__ == "__main__":
        main()