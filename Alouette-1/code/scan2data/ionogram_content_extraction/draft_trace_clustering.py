# -*- coding: utf-8 -*-
"""
Starting code to extract the difference subtraces in an ionogram trace

"""

#Library imports
import random
import glob
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from extract_all_coordinates_ionogram_trace import extract_coord,map_coordinates_positions_to_values
sys.path.append('../')
from image_segmentation.segment_images_in_subdir import segment_images
from ionogram_grid_determination.grid_mapping import all_stack,get_grid_mappings

def two_clusters(sorted_array):
    """ Cluster sorted array into 2 subarray in a way to minimize the sum of variances
    
    :param sorted_array: sorted array to cluster
    :type sorted_array: class: `numpy.ndarray`
    :returns: (subarray1, subarray2): the two split class: `numpy.ndarray`
    :rtype: (class: `numpy.ndarray`,class: `numpy.ndarray`)
    
    """
    if len(sorted_array) < 3:
        return (sorted_array, sorted_array)
    variances = [np.std(sorted_array[0:i])+np.std(sorted_array[i:]) for i in range(1, len(sorted_array)-1)]
    index_sep = np.argmin(variances) + 1
    subarray1, subarray2 = sorted_array[0:index_sep ],sorted_array[index_sep:]

    return (subarray1, subarray2)

def subtraces_extraction(arr_coord, 
                         cutoff_outlier=10 ):
    """Obtain the subtraces using clustering based on median values
    
    :param arr_coord: coordinates of an ionogram 
    :type arr_coord: class: `numpy.ndarray
    :param cutoff_outlier: ,defaults to 10
    :type cutoff_outlier: int, optional
    :returns: df_arr_coord : Dataframes containing the subtraces
    :rtype: class: `pandas.core.frame.DataFrame
    """
    # Sort arr_coord
    sorted_arr_coord = sorted(arr_coord , key=lambda coord: coord[0])
    sorted_x, sorted_y = zip(*sorted_arr_coord)
    
    # Split arr_coord by unique frequency values
    x_ukeys, x_index = np.unique(sorted_x, return_index=True)
    y_arrays = np.split(sorted_y, x_index[1:])
    df_arr_coord = pd.DataFrame({'Hz':x_ukeys, 'Km': y_arrays}  )
        
    # Determine median values and remove outliers
    df_arr_coord['Median'] = df_arr_coord.apply(lambda row: np.median(row['Km']) ,1)
    df_arr_coord['Median_diff'] = df_arr_coord.apply(lambda row: abs(row['Km'] - row['Median']) ,1)
    df_arr_coord['MAD'] = df_arr_coord.apply(lambda row: np.median(row['Median_diff']) ,1)
    df_arr_coord['Absolute_dev_median'] = df_arr_coord.apply(lambda row: row['Median_diff']/row['MAD'] ,1)
    df_arr_coord['Km_no_outliers'] = df_arr_coord.apply(lambda row: row['Km'][row['Absolute_dev_median'] <= cutoff_outlier] ,1)

    # Cluster ionogram trace into subtraces
    df_arr_coord['Cluster1'],df_arr_coord['Cluster2'] = zip(*df_arr_coord['Km_no_outliers'].map(two_clusters))
    df_arr_coord['Cluster1_median'] = df_arr_coord.apply(lambda row: np.median(row['Cluster1']) ,1)
    df_arr_coord['Cluster2_median'] = df_arr_coord.apply(lambda row: np.median(row['Cluster2']) ,1)
    df_arr_coord = df_arr_coord[['Hz', 'Km', 'Median', 'Cluster1', 'Cluster2','Cluster1_median','Cluster2_median']]

    return df_arr_coord

def plot_curve_extraction(regex_subdir,regex_images,to_overlay):
    """Plot the output of subtraces_extraction on a random ionogram
    
    :param regex_subdir: regular expression to extract subdirectory ex: 'E:/master/R*/[0-9]*/'
    :type regex_img: str
    :param regex_img: regular expression to extract images ex: '*.png'
    :type regex_img: str
    :param to_overlay: whether to plot on raw ionogram
    :type to_overlay: bool
    
    """
    # All the subdirectory i.e. R014207948/1743-9/
    list_all_subdir = glob.glob(regex_subdir)
    
    # Randomly pick a subdirectory
    sample_subdir = list_all_subdir[random.randint(0,len(list_all_subdir) - 1)]
    
    # Segment images in the subdirectory
    df_img,_,_ =segment_images(sample_subdir, regex_images)

    # Get stack
    stack = all_stack(df_img)
    col_peaks,row_peaks,mapping_Hz, mapping_km = get_grid_mappings(stack)
    
    # Get random ionogram 
    df_sample = df_img.sample(n=1) 
    ionogram = df_sample['ionogram'].iloc[0]
    
    # Get coordinates of trace
    raw_coord, window_coord = extract_coord(ionogram ,col_peaks,row_peaks)
    
    # Plot clusterings
    fig,axes = plt.subplots(nrows=2,ncols=2)
    fig.suptitle(df_sample['file_name'].iloc[0])
    ax= axes.ravel()
    ax[0].set_title("Original ionogram " )
    ax[0].imshow(ionogram,'gray')
    ax[3].set_title("Clustered extracted trace" )
    if to_overlay:
        ax[1].set_title("Extracted trace " )
        ax[1].imshow(ionogram,'gray')
        ax[1].scatter(list(zip(*raw_coord))[0],list(zip(*raw_coord))[1],s=1)
        ax[2].set_title("Windowed extracted trace " )
        ax[2].imshow(ionogram,'gray')
        ax[2].scatter(list(zip(*window_coord))[0],list(zip(*window_coord))[1],s=1)

        ax[3].imshow(ionogram,'gray')
        df_arr_raw_coord = subtraces_extraction(window_coord )
        ax[3].scatter(list(zip(*window_coord))[0],list(zip(*window_coord))[1],s=1, label='raw')
        ax[3].scatter(df_arr_raw_coord['Hz'].tolist(),df_arr_raw_coord['Median'].tolist(),s=5, label='median')
        ax[3].scatter(df_arr_raw_coord ['Hz'].tolist(),df_arr_raw_coord['Cluster1_median'].tolist(),s=5,label='median1')
        ax[3].scatter(df_arr_raw_coord ['Hz'].tolist(),df_arr_raw_coord['Cluster2_median'].tolist(),s=5,label='median2') 
        for i in range(0,4):
            ax[i].axis('off')
    else:
        ax[1].set_title("Windowed extracted trace " )
        ax[1].scatter(list(zip(*window_coord))[0],list(zip(*window_coord))[1],s=1)
        ax[1].invert_yaxis()
        arr_adjusted_coord = map_coordinates_positions_to_values(window_coord,col_peaks,row_peaks,mapping_Hz,mapping_km)
        ax[2].set_title("Labeled extracted trace " )
        ax[2].scatter(list(zip(*arr_adjusted_coord))[0],list(zip(*arr_adjusted_coord))[1],s=1)
        
        
        df_arr_adjusted_coord = subtraces_extraction(arr_adjusted_coord)
        ax[3].scatter(list(zip(*arr_adjusted_coord ))[0],list(zip(*arr_adjusted_coord ))[1],s=1, label='raw')
        ax[3].scatter(df_arr_adjusted_coord['Hz'].tolist(),df_arr_adjusted_coord['Median'].tolist(),s=5, label='median')
        ax[3].scatter(df_arr_adjusted_coord ['Hz'].tolist(),df_arr_adjusted_coord['Cluster1_median'].tolist(),s=5,label='median1')
        ax[3].scatter(df_arr_adjusted_coord ['Hz'].tolist(),df_arr_adjusted_coord['Cluster2_median'].tolist(),s=5,label='median2') 
        for i in range(0,2):
            ax[i].axis('off')
        for i in range(2,4):
            ax[i].invert_yaxis()
            ax[i].set_xlabel('Frequency (Hz)')
            ax[i].set_ylabel('Depth (km)')
    
    plt.legend(loc='best')

        
if __name__ == '__main__':
    plot_curve_extraction(regex_subdir='E:/master/R*/[0-9]*/',regex_images='*.png',to_overlay=True)
    plot_curve_extraction(regex_subdir='E:/master/R*/[0-9]*/',regex_images='*.png',to_overlay=False)

    
