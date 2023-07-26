# -*- coding: utf-8 -*-
"""
Test for scripts in segment_images_in_subdir.py
"""
# Library imports
import sys

import numpy as np
import matplotlib.pyplot as plt
import cv2

from extract_ionogram_from_scan import extract_ionogram
from extract_metadata_from_scan import extract_metadata
from segment_images_in_subdir import segment_images, LIST_FLIP_VERTICAL, LIST_ROTATE_180
from trim_raw_metadata import bottomside_metadata_trimming,leftside_metadata_trimming,connected_components_metadata_location

sys.path.append('../')
from helper_functions import generate_random_subdirectory,generate_random_image_from_subdirectory


def test_segment_images_in_subdir(sample_subdirectory,sample_image_path, regex_images):
    """Visualize the output of segment_images_in_subdir.segment_images on a random subdirectory
    
    :param sample_subdirectory: sample subdir to test segment_images_in_subdir
    :type sample_subdirectory: str
    :param regex_images: regular expression to extract images ex: '*.png'
    :type regex_images: str
    """ 
    # Run segment_images on the subdirectory
    df_img,df_loss,df_outlier =segment_images(sample_subdir, regex_images)
    
    # Visualize a random row of the dataframe output of df_img
    df_sample = df_img[df_img['file_name'] == sample_image_path]
    row = df_sample.iloc[0,:]
    
    fig,axes = plt.subplots(nrows=2,ncols=2)
    ax=axes.ravel()
    fig.suptitle(row['file_name'])
    ax[0].imshow(row['raw'],'gray')
    ax[1].imshow(row['raw_metadata'],'gray')
    ax[2].imshow(row['ionogram'],'gray')
    ax[3].imshow(row['trimmed_metadata'],'gray')
    fig.suptitle(row['file_name'])
    ax[0].set_title('raw')
    ax[1].set_title('raw_metadata')
    ax[2].set_title('ionogram')
    ax[3].set_title('trimmed_metadata')

def test_segment_images_in_subdir_extract_helpers(sample_image_path):
    """Visualize the output of extraction helpers of segment_images_in_subdir.segment_images on a random image
    including extract_ionogram_from_scan.extract_ionogram and extract_metadata_from_scan.extract_metadata
    
    :param sample_image_path: path of the sample image
    :type sample_image_path: str
    """

    """
    Following Code pasted from segment_images_in_subdir.segment_images
    """
    
    # If flipping/rotating is required 
    path = sample_image_path.replace('\\', '/')
    flip_vertical = any([subdir_dir in path for subdir_dir in LIST_FLIP_VERTICAL])
    rotate_180 = any([subdir_dir in path for subdir_dir in LIST_ROTATE_180])
 
    #Read image in an 2D UTF-8 grayscale array
    if flip_vertical == True:
        raw_img = cv2.flip(cv2.imread(sample_image_path,0),1)
    else:
        raw_img = cv2.imread(sample_image_path,0)
    
    # Extract ionogram and coordinates delimiting its limits
    limits,ionogram= extract_ionogram(raw_img)
    
    
    if rotate_180 == True:
        ionogram= np.rot90(ionogram, 2)

    # Raw metadata
    metadata_type,raw_metadata = extract_metadata(raw_img, limits)
    if rotate_180 == True:
        raw_metadata =  np.rot90(raw_metadata, 2)
    
    """
    Above Code pasted from segment_images_in_subdir.segment_images
    """
    fig,axes = plt.subplots(nrows=3)
    ax=axes.ravel()
    fig.suptitle(sample_image_path)
    ax[0].imshow(raw_img,'gray')
    ax[1].imshow(raw_metadata,'gray')
    ax[2].imshow(ionogram,'gray')

    fig.suptitle(sample_image_path)
    ax[0].set_title('raw')
    ax[1].set_title('raw_metadata')
    ax[2].set_title('ionogram')


def test_segment_images_in_subdir_trim_helper(sample_image_path):
    """Visualize the output of functions of extract_metadata_from_scan.trim_raw_metadata on a random image, a helper script of segment_images_in_subdir.segment_images 
    
    :param sample_image_path: path of the sample image
    :type sample_image_path: str
    """

    """
    Following Code heavily inspired from segment_images_in_subdir.segment_images
    """
    
    # If flipping/rotating is required 
    path = sample_image_path.replace('\\', '/')
    flip_vertical = any([subdir_dir in path for subdir_dir in LIST_FLIP_VERTICAL])
    rotate_180 = any([subdir_dir in path for subdir_dir in LIST_ROTATE_180])
 
    #Read image in an 2D UTF-8 grayscale array
    if flip_vertical == True:
        raw_img = cv2.flip(cv2.imread(sample_image_path,0),1)
    else:
        raw_img = cv2.imread(sample_image_path,0)
    
    # Extract ionogram and coordinates delimiting its limits
    limits,ionogram= extract_ionogram(raw_img)
    
    
    if rotate_180 == True:
        ionogram= np.rot90(ionogram, 2)

    # Raw metadata
    metadata_type,raw_metadata = extract_metadata(raw_img, limits)
    if rotate_180 == True:
        raw_metadata =  np.rot90(raw_metadata, 2)
    
    """
    Above Code heavily inspired from segment_images_in_subdir.segment_images
    """
    median_kernel_size=5
    opening_kernel_size = (3,3)
    """
    Following Code pasted from segment_images_in_subdir.trim_raw_metadata
    """
    # Median filtering to remove salt and pepper noise
    median_filtered_meta = cv2.medianBlur(raw_metadata,median_kernel_size)
    
    # Opening operation: Erosion + Dilation
    kernel_opening = np.ones(opening_kernel_size,np.uint8)
    opened_meta = cv2.morphologyEx(median_filtered_meta,cv2.MORPH_OPEN,kernel_opening)
    
    # Binarizatinon for connected component algorithm
    _,meta_binary = cv2.threshold(opened_meta, 127,255,cv2.THRESH_BINARY)    
    
    # Run connected component algorithm
    connected_meta = connected_components_metadata_location(meta_binary)
    
    if metadata_type == 'left':
        trimmed_metadata = leftside_metadata_trimming(connected_meta,meta_binary)
    else:
        trimmed_metadata =  bottomside_metadata_trimming(connected_meta,opened_meta)
    
    """
    Above Code pasted from segment_images_in_subdir.trim_raw_metadata
    """
    #Label the image for display        
    label_hue = np.uint8(179*connected_meta/np.max(connected_meta))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue==0] = 0
    
    
    fig,axes = plt.subplots(ncols=2,nrows=3)
    ax=axes.ravel()

    ax[0].imshow(raw_img,'gray')
    ax[1].imshow(median_filtered_meta,'gray')
    ax[2].imshow(opened_meta,'gray')
    ax[3].imshow(meta_binary,'gray')
    ax[4].imshow(labeled_img)
    ax[5].imshow(trimmed_metadata,'gray')
    
    fig.suptitle(sample_image_path)
    ax[0].set_title('raw')
    ax[1].set_title('median_filtered_meta')
    ax[2].set_title('opened_meta')
    ax[3].set_title('meta_binary')
    ax[4].set_title('connected_meta')
    ax[5].set_title('trimmed_metadata' + metadata_type)
    
if __name__ =='__main__':
    sample_subdir = generate_random_subdirectory(regex_subdirectory='L:/DATA/ISIS/raw_upload_20230421/R*/B*/')
    sample_img_from_subdir = generate_random_image_from_subdirectory(sample_subdir, regex_images='*.png')
    
    test_segment_images_in_subdir(sample_subdir,sample_img_from_subdir, regex_images='*.png')
    test_segment_images_in_subdir_extract_helpers(sample_img_from_subdir)
    test_segment_images_in_subdir_trim_helper(sample_img_from_subdir)
    