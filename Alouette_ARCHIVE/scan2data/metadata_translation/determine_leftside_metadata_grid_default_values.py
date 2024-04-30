# -*- coding: utf-8 -*-
"""
Determine default values for the grid to determine metadata values

"""

#Library imports
import glob
import sys
import traceback
from itertools import chain

import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from leftside_metadata_grid_mapping import indices_highest_peaks_hist_binning
from translate_leftside_metadata import extract_centroids_and_determine_type
sys.path.append('../')
from helper_functions import record_loss
from image_segmentation.segment_images_in_subdir import segment_images


def get_peaks(list_coord):
    """From a list of coordinates, return a list of the most common values through binning
    
    :param list_coord: list of positions where metadata is detected
    :type list_coord: class: `list`
    :returns: peaks i.e. list of the most common values through binning
    :rtype: class: `numpy.ndarray`
    
    
    
    """
    idx_peaks,bin_edges,counts = indices_highest_peaks_hist_binning(list_coord)
    return bin_edges[np.array(idx_peaks)]



def get_leftside_metadata_grid_peaks(regex_subdir, regex_images,
                                     min_subset=10):
    """Generates a dataframe containing information about peaks to generate metadata grids
    
    :param regex_subdir: regular expression to extract subdirectory ex: 'E:/master/R*/[0-9]*/'
    :type regex_img: str
    :param regex_img: regular expression to extract images ex: '*.png'
    :type regex_img: str
    :param min_subset: minimum number of items extracted to be considered ,defaults to 10
    :type min_subset: int, optional
    :returns: df_summary_left_dot,df_summary_left_num,error_list: Dataframe listing peaks for dot metadata located on the left of ionogram,  Dataframe listing peaks for number metadata located on the left of ionogram, list of filenames leading to errors
    :rtype: `pandas.core.frame.DataFrame`,`pandas.core.frame.DataFrame`, list
    
    """
    # All the subdirectory i.e. R014207948/1743-9/
    list_all_subdir = glob.glob(regex_subdir)

    df_summary_left_dot = pd.DataFrame(columns=['meta_peaks_x','meta_peaks_y'] )
    df_summary_left_num = pd.DataFrame(columns=['meta_peaks_x','meta_peaks_y'] )
    error_list = []
    
    
    for i,subdir_name in enumerate(list_all_subdir):
        try:
            df_img,_,_ =segment_images(subdir_name, regex_images)
            df_img_subset = df_img[df_img['metadata_type']=='left']
            if i % 50 == 0:
                print(i)
            
            if len(df_img_subset.index) > min_subset + 1:
            
                '''
                Following Code pasted from translate_leftside_metadata.get_leftside_metadata
                '''
                # Centroids extraction
                df_img_subset['rotated_metadata'] = df_img_subset['trimmed_metadata'].map(lambda trimmed_meta: np.rot90(trimmed_meta,-1))
                kernel_dilation = np.ones((1,1),np.uint8)
                df_img_subset['dilated_metadata'] = df_img_subset['rotated_metadata'].map(lambda rotated_meta: cv2.dilate(rotated_meta,kernel_dilation))
                df_img_subset['x_centroids'],df_img_subset['y_centroids'],df_img_subset['is_dot'] = zip(*df_img_subset.apply(lambda row: extract_centroids_and_determine_type(row['dilated_metadata'],row['file_name']),1))
                _,loss_centroids_extraction = record_loss(df_img_subset,'metadata_translation.determine_leftside_metadata_grid_mapping.extract_centroids_and_determine_type',subdir_name)
                  
                # Remove files whose centroid metadata extraction was not successful
                df_img_subset = df_img_subset[~loss_centroids_extraction]
                
                '''
                Above Code pasted from translate_leftside_metadata.get_leftside_metadata
                '''
                # Determine metadata mapping for dot-type metadata and num-type metadata
                df_dot_subset = df_img_subset[df_img_subset['is_dot'] == True]
                df_num_subset = df_img_subset[df_img_subset['is_dot'] == False]
                

                list_x_dot, list_y_dot,list_x_digit,list_y_digit = [0],[0],[0],[0]
                if not df_dot_subset.empty:
                    list_x_dot = list(chain(*df_dot_subset['x_centroids'].tolist()))
                    list_y_dot = list(chain(*df_dot_subset['y_centroids'].tolist()))
                    x_peaks_dot,y_peaks_dot = get_peaks(list_x_dot),get_peaks(list_y_dot )
                    to_apend_dot = pd.DataFrame({'meta_peaks_x':[x_peaks_dot],'meta_peaks_y':[y_peaks_dot]} )
                    df_summary_left_dot = df_summary_left_dot.append(to_apend_dot)
                
                if not df_num_subset.empty:
                    list_x_digit = list(chain(*df_num_subset['x_centroids'].tolist()))
                    list_y_digit = list(chain(*df_num_subset['y_centroids'].tolist()))
                    x_peaks_num,y_peaks_num = get_peaks(list_x_digit),get_peaks(list_y_digit )
                    to_apend_num = pd.DataFrame({'meta_peaks_x':[x_peaks_num],'meta_peaks_y':[y_peaks_num]} )
                    df_summary_left_num = df_summary_left_num.append(to_apend_num)

        except Exception:
            traceback.print_exc()
            error_list.append(subdir_name)
            print(subdir_name )
            
    return df_summary_left_dot,df_summary_left_num,error_list



def  plot_hist_peaks_grids(*all_df,
                          nbins=500):
    """Plots histogram to determine default values for the ionogram grids
    
    :param *all_df: dataframes (from grid_default_values) whose values are to be plotted
    :type *all_df: tuple of or single class: `pandas.core.frame.DataFrame`
    :param nbins: number of bins used for histogram, defaults to 500
    :type nbins: int, optional
    """
    nrow = len(*all_df)
    fig,axes = plt.subplots(nrows=nrow,ncols=2)
    ax = axes.ravel()

    for i,df in enumerate(*all_df):
        peaks_x = list(chain.from_iterable(df['meta_peaks_x']))
        select_peaks_idx,bin_edges,counts = indices_highest_peaks_hist_binning(peaks_x)
        bin_centers = (0.5*(bin_edges[1:] + bin_edges[:-1]))
        peaks = bin_edges[np.array(select_peaks_idx)]
        ax[2*i].plot(bin_centers,counts)
        ax[2*i].plot(peaks,counts[select_peaks_idx], "x")
        print(peaks)
        ax[2*i].set_title('meta_peaks_x')
        
        peaks_y = list(chain.from_iterable(df['meta_peaks_y']))
        select_peaks_idx,bin_edges,counts = indices_highest_peaks_hist_binning(peaks_y)
        bin_centers = (0.5*(bin_edges[1:] + bin_edges[:-1]))
        peaks = bin_edges[np.array(select_peaks_idx)]
        ax[2*i+1].plot(bin_centers,counts)
        ax[2*i+1].plot(peaks,counts[select_peaks_idx], "x")
        print(peaks)
        ax[2*i+1].set_title('meta_peaks_y')


if __name__ == '__main__':
    df_summary_left_dot1,df_summary_left_num1,error_list1 = get_leftside_metadata_grid_peaks(regex_subdir='E:/master/R*/[0-9]*/', regex_images='*.png')
    df_summary_left_dot2,df_summary_left_num2,error_list2 = get_leftside_metadata_grid_peaks(regex_subdir='G:/AlouetteData/Alouette Data/R*/[0-9]*[0-9]/', regex_images='Image*[0-9].png')
    plot_hist_peaks_grids((df_summary_left_dot1,df_summary_left_num1))
    plot_hist_peaks_grids((df_summary_left_dot2,df_summary_left_num2))
#mean_dist_default for DEFAULT_DICT_NUM_DIGIT:  print(np.mean([np.array([ 40.60056,85.560816,137.546112,182.506368])[i+1] - num for i, num in enumerate(np.array([ 40.60056,85.560816,137.546112,182.506368])[:-1])]))
# mean_dist_default for DEFAULT_DICT_NUM_DOT: print(np.mean([np.array([31.646912, 45.832512, 59.7344  , 77.891968])[i+1] - num for i, num in enumerate(np.array([31.646912, 45.832512, 59.7344  , 77.891968])[:-1])]))



