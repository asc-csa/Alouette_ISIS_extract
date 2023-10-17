# -*- coding: utf-8 -*-
"""
Test functions in ionogram content extraction
"""

#Library imports
import random
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np

from extract_all_coordinates_ionogram_trace import extract_coord,map_coordinates_positions_to_values, extract_coord_subdir_and_param
from extract_select_parameters import extract_fmin_and_max_depth

sys.path.append('../')
from image_segmentation.segment_images_in_subdir import segment_images
from ionogram_grid_determination.grid_mapping import all_stack,get_grid_mappings
from helper_functions import generate_random_subdirectory,generate_random_image_from_subdirectory

def test_coord_extraction(sample_subdirectory,sample_image_path,regex_images,to_overlay):
    """ Plot the output of extract_all_coordinates_ionogram_trace.extract_coord and extract_all_coordinates_ionogram_trace.map_coordinates_positions_to_values on a random ionogram
    
    :param sample_subdirectory: sample subdir to test coord_extraction and map_coordinates_positions_to_values
    :type sample_subdirectory: str
    :param sample_image_path: path of the sample image
    :type sample_image_path: str
    :param regex_images: regular expression to extract images ex:'E:/master/R*/[0-9]*/'
    :type regex_images: str
    :param to_overlay: whether to plot on raw ionogram
    :type to_overlay: bool
    """

    # Segment images in the subdirectory
    df_img,_,_ =segment_images(sample_subdirectory, regex_images)

    # Get stack
    stack = all_stack(df_img)
    col_peaks,row_peaks,mapping_Hz, mapping_km = get_grid_mappings(stack)
    
    # Get data from select raw image
    df_sample = df_img[df_img['file_name'] == sample_image_path]
    row = df_sample.iloc[0,:]

    ionogram = row['ionogram']
    
    # Get coordinates of trace
    raw_coord, window_coord = extract_coord(ionogram ,col_peaks,row_peaks)
    
    # (Hz, km) coordinates
    arr_adjusted_coord = map_coordinates_positions_to_values(window_coord,col_peaks,row_peaks,mapping_Hz,mapping_km)
    
    # Plots
    fig,axes = plt.subplots(nrows=2,ncols=2)
    ax= axes.ravel()
    fig.suptitle(row['file_name'])
    ax[0].set_title("Original ionogram " )
    ax[0].imshow(ionogram,'gray')
    if to_overlay:
        ax[1].set_title("Extracted trace " )
        ax[1].imshow(ionogram,'gray')
        ax[1].scatter(list(zip(*raw_coord))[0],list(zip(*raw_coord))[1],s=1)
        
        ax[2].set_title("Windowed extracted trace " )
        ax[2].imshow(ionogram,'gray')
        ax[2].scatter(list(zip(*window_coord))[0],list(zip(*window_coord))[1],s=1)

        for i in range(3):
            ax[i].axis('off')
    else:
        ax[1].set_title("Extracted trace " )
        ax[1].scatter(list(zip(*raw_coord))[0],list(zip(*raw_coord))[1],s=1)
        ax[1].invert_yaxis()
        
        ax[2].set_title("Windowed extracted trace " )
        ax[2].scatter(list(zip(*window_coord))[0],list(zip(*window_coord))[1],s=1)
        ax[2].invert_yaxis()

    for i in range(3):
        ax[i].axis('off')
    ax[3].set_title("Labeled extracted trace " )
    ax[3].scatter(list(zip(*arr_adjusted_coord))[0],list(zip(*arr_adjusted_coord))[1],s=1)
            
    ax[3].invert_yaxis()
    ax[3].set_xlabel('Frequency (Hz)')
    ax[3].set_ylabel('Depth (km)')
    


def test_param_extraction(sample_subdirectory,sample_image_path,regex_images):
    """ Plot the output of extract_select_parameters.extract_fmin_and_max_depth
    
    :param sample_subdirectory: sample subdir to test coord_extraction and map_coordinates_positions_to_values
    :type sample_subdirectory: str
    :param sample_image_path: path of the sample image
    :type sample_image_path: str
    :param regex_images: regular expression to extract images ex:'E:/master/R*/[0-9]*/'
    :type regex_images: str

    """
    
    # Segment images in the subdirectory
    df_img,_,_ =segment_images(sample_subdirectory, regex_images)

    # Get stack
    stack = all_stack(df_img)
    col_peaks,row_peaks,mapping_Hz, mapping_km = get_grid_mappings(stack)

    # Get random ionogram 
    df_sample = df_img[df_img['file_name'] == sample_image_path]
    row = df_sample.iloc[0,:]

    ionogram = row['ionogram']

    # Get (x,y) coordinates of trace
    raw_coord, window_coord = extract_coord(ionogram ,col_peaks,row_peaks)

    # (Hz, km) coordinates
    arr_adjusted_coord = map_coordinates_positions_to_values(window_coord,col_peaks,row_peaks,mapping_Hz,mapping_km)  

    # Extract parameters
    fmin, depth_max = extract_fmin_and_max_depth(arr_adjusted_coord)

    # For plotting purposes
    # remove outliers ie coordinates less coordinates corresponding to 0.5 Hz or more than corresponding to 11.5 Hz
    col_peaks = np.array(list(mapping_Hz.values())) # use the modified col_peaks ie the one with exactly 13 values
    mask = np.logical_or(window_coord[:,0] < col_peaks.min(), window_coord[:,0] > col_peaks.max())
    window_coord_adjusted = window_coord[~mask,:]

    fmin_coord, depth_max_coord = extract_fmin_and_max_depth(window_coord_adjusted,if_raw=True)

    # Plots
    fig,axes = plt.subplots(nrows=1,ncols=2)
    ax= axes.ravel()
    fig.suptitle(row['file_name'])
    ax[0].set_title("Original ionogram " )
    ax[0].imshow(ionogram,'gray')

    colored_iono = cv2.cvtColor(ionogram ,cv2.COLOR_GRAY2RGB)
    h,w = ionogram.shape
    cv2.line(colored_iono,(int(fmin_coord),0),(int(fmin_coord),h),[0,0,255],5)
    cv2.line(colored_iono,(0,int(depth_max_coord)),(w,int(depth_max_coord)),[0,0,255],5)
    ax[1].imshow(colored_iono)
    ax[1].set_title(f'fmin={str(fmin)}max_depth={str(depth_max)}')
    plt.axis('off')
    
    
def test_extract_coord_subdir_and_param(sample_subdirectory,sample_image_path,regex_images,to_overlay):
    """ Plot the output of extract_all_coordinates_ionogram_trace.extract_coord_subdir_and_param
    
    :param sample_subdirectory: sample subdir to test coord_extraction and map_coordinates_positions_to_values
    :type sample_subdirectory: str
    :param sample_image_path: path of the sample image
    :type sample_image_path: str
    :param regex_images: regular expression to extract images ex:'E:/master/R*/[0-9]*/'
    :type regex_images: str
    :param to_overlay: whether to plot on raw ionogram
    :type to_overlay: bool
    
    """
    
    # Segment images in the subdirectory
    df_img,_,_ =segment_images(sample_subdirectory, regex_images)

    # Get stack
    stack = all_stack(df_img)
    col_peaks,row_peaks,mapping_Hz, mapping_km = get_grid_mappings(stack)

    df_img, df_loss_coord = extract_coord_subdir_and_param(df_img,sample_subdirectory,col_peaks,row_peaks,mapping_Hz,mapping_km)

    # Visualize a random row of the dataframe output of df_img
    df_sample = df_img[df_img['file_name'] == sample_image_path]
    row = df_sample.iloc[0,:]


    fig,axes = plt.subplots(nrows=2,ncols=2)
    ax=axes.ravel()
    ax[0].set_title("Original ionogram " )
    ax[0].imshow(row['ionogram'],'gray')
    if to_overlay:
        ax[1].set_title("Windowed extracted trace " )
        ax[1].imshow(row['ionogram'],'gray')
        ax[1].scatter(list(zip(*row['window_coord']))[0],list(zip(*row['window_coord']))[1],s=1)

    else:
        ax[1].set_title("Windowed extracted trace " )
        ax[1].scatter(list(zip(*row['window_coord']))[0],list(zip(*row['window_coord']))[1],s=1)
        ax[1].invert_yaxis()

    colored_iono = cv2.cvtColor(row['ionogram'] ,cv2.COLOR_GRAY2RGB)
    h,w = row['ionogram'].shape

    col_peaks = np.array(list(mapping_Hz.values())) # use the modified col_peaks ie the one with exactly 13 values
    mask = np.logical_or(row['window_coord'][:,0] < col_peaks.min(), row['window_coord'][:,0] > col_peaks.max())
    window_coord_adjusted = row['window_coord'][~mask,:]

    fmin_coord, depth_max_coord = extract_fmin_and_max_depth(window_coord_adjusted,if_raw=True)

    cv2.line(colored_iono,(int(fmin_coord),0),(int(fmin_coord),h),[0,0,255],5)
    cv2.line(colored_iono,(0,int(depth_max_coord)),(w,int(depth_max_coord)),[0,0,255],5)
    ax[2].imshow(colored_iono)
    ax[2].set_title('fmin='+  str(row['fmin']) + 'max_depth='+  str(row['max_depth']))

    for i in range(3):
        ax[i].axis('off')

    ax[3].set_title("Labeled extracted trace " )
    ax[3].scatter(list(zip(*row['mapped_coord']))[0],list(zip(*row['mapped_coord']))[1],s=1)

    ax[3].invert_yaxis()
    ax[3].set_xlabel('Frequency (Hz)')
    ax[3].set_ylabel('Depth (km)')
    

    
if __name__ == '__main__':
    
    sample_subdir = generate_random_subdirectory(regex_subdirectory='L:/DATA/ISIS/raw_upload_20230421/R014207869/B*/')
    sample_img_from_subdir = generate_random_image_from_subdirectory(sample_subdir, regex_images='*.png')
        
    test_coord_extraction(sample_subdir,sample_img_from_subdir,regex_images='*.png',to_overlay=True)
    test_coord_extraction(sample_subdir,sample_img_from_subdir,regex_images='*.png',to_overlay=False)
    test_param_extraction(sample_subdir,sample_img_from_subdir,regex_images='*.png')
        
    test_extract_coord_subdir_and_param(sample_subdir,sample_img_from_subdir,regex_images='*.png',to_overlay=True)
    test_extract_coord_subdir_and_param(sample_subdir,sample_img_from_subdir,regex_images='*.png',to_overlay=False)
    
