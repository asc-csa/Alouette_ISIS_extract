# -*- coding: utf-8 -*-
"""
Interact with user input through command line

"""
import glob
import ntpath
import numpy as np
import process_directory
import sys

import warnings
warnings.filterwarnings('ignore')


def main():
    processAllSubdirectories = True #will not pass to process-extract-management
    master_dir = sys.argv[1] #'C:/Users/rnaidoo/Documents/Projects_data/Alouette_I/01_intake/' #input('Type directory with all the raw data ex: E:/master/: ')
    dir_csv_output = sys.argv[2] #'C:/Users/rnaidoo/Documents/Projects_data/Alouette_I/02_result/' #input('Type directory for outputs of processing ex: C:/Users/JPatel/Desktop/: ')
    regex_raw = '*.png'
    desire_pickle = False #enter True if you want to save it as a pickle file.
    desire_only_leftside = False #Enter False for it to analyze the bottom data.

    if processAllSubdirectories:
        list_all_subdir = glob.glob(master_dir + 'R*/[0-9]*/')
        for sample_subdir in list_all_subdir:
            sample_subdir = sample_subdir.replace('/', '\\')
            sample_subdir = sample_subdir.replace('\\', '/')
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
            