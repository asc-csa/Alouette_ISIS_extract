# -*- coding: utf-8 -*-
"""
Test for scripts in segment_images_in_subdir.py

"""
# Library imports
import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt

from grid_mapping import all_stack,get_grid_mappings
sys.path.append('../')
from image_segmentation.segment_images_in_subdir import segment_images
from helper_functions import generate_random_subdirectory

def test_grid_mapping(sample_subdirectory, regex_images):
    """Visualize the output of grid_mapping.get_grid_mappings on a random subdirectory
    
    :param sample_subdirectory: sample subdir to test get_grid_mappings
    :type sample_subdirectory: str
    :param regex_images: regular expression to extract images ex: '*.png'
    :type regex_images: str
    """
    

    # Segment images in the subdirectory
    df_img,_,_ =segment_images(sample_subdirectory, regex_images)

    # Get stack
    stack = all_stack(df_img)
    col_peaks,row_peaks,mapping_Hz, mapping_km = get_grid_mappings(stack)
            
    fig,axes = plt.subplots(ncols=2)
    ax = axes.ravel()
                
    # Plot stack
    fig.suptitle(sample_subdirectory)
    ax[0].imshow(stack,'gray')
    h,w = stack.shape
    grid = np.ones((h,w),np.uint8)
    col_peaks2 = np.asarray(list(mapping_Hz.values()))
    
    for i in col_peaks2:
        cv2.line(grid , (i, 0), (i,h), 0, 5, 1)
    for i in row_peaks:
        cv2.line(grid , (0, i), (w,i), 0, 5, 1)
    ax[1].imshow(grid, 'gray')
    


if __name__ == '__main__':
    sample_subdir = generate_random_subdirectory(regex_subdirectory='L:/DATA/ISIS/raw_upload_20230421/R*/B*/')
    test_grid_mapping(sample_subdir , regex_images='*.png')
    