# -*- coding: utf-8 -*-
"""

"""
import glob
import process_directory_md
import sys

import warnings
warnings.filterwarnings('ignore')


def main():
    master_dir = sys.argv[1] #03_processing
    dir_csv_output = sys.argv[2] #05a_result_local
    regex_raw = '*.png'

    list_all_subdir = glob.glob(f'{master_dir}R*/[0-9]*/')
    for sample_subdir in list_all_subdir:
        sample_subdir = sample_subdir.replace('/', '\\')
        sample_subdir = sample_subdir.replace('\\', '/')
        process_directory_md.process_extract_management(dir_csv_output, master_dir, regex_raw, sample_subdir)


if __name__ == "__main__":
        main()
            
