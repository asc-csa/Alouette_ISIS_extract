# -*- coding: utf-8 -*-
"""
Determine default values for the grid in ionogram plots
"""

# Library imports
import glob
import traceback
import itertools
import sys

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from grid_mapping import all_stack,get_grid_mappings
sys.path.append('../')
from image_segmentation.segment_images_in_subdir import segment_images
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

def plot_hist_peaks_grids(*all_df,
                          nbins=500):
    '''Plots histogram to determine default values for the ionogram grids
    
    :param *all_df: dataframes (from grid_default_values) whose values are to be plotted
    :type *all_df: tuple of or single class: `pandas.core.frame.DataFrame`
    :param nbins: number of bins used for histogram, defaults to 500
    :type nbins: int, optional
    
    
    '''
    nrow = len(all_df)
    fig,axes = plt.subplots(nrows=nrow,ncols=2)
    ax = axes.ravel()
    
    for i,df in enumerate(all_df):
        coord_row = list(itertools.chain.from_iterable(df['row_peaks']))
        
        
        ax[2*i].hist(coord_row, bins=nbins)
        ax[2*i].set_title(df.name + ' ' + 'row_peaks')
        
        coord_col = list(itertools.chain.from_iterable(df['col_peaks']))
        ax[2*i +1].hist(coord_col, bins=nbins)
        ax[2*i+1].set_title(df.name + ' '+ 'col_peaks')
        
        list_dict_mapping_Hz = df['mapping_Hz'].tolist()
        
    
     
    for i,df in enumerate(all_df):
        fig2,axes2 = plt.subplots(nrows=4,ncols=4)
        ax2 = axes2.ravel()
        fig2.suptitle(df.name)
        i = 0
        list_dict_mapping_Hz = df['mapping_Hz'].tolist()
        # real mapping
        #list_dict_mapping_Hz = [{dict_mapping[k] : k for k in dict_mapping} for dict_mapping in list_dict_mapping_Hz]
        combined_mapping_Hz = {1.5:[], 2.0:[], 2.5:[], 3.5: [], 4.5: [], 5.5: [], 6.5: [], 7.0:[],7.5:[],8.5:[],9.5:[],10.5:[],11.5:[]}
        for key in combined_mapping_Hz:
            combined_mapping_Hz[key] = [dict_mapping[key] for dict_mapping in list_dict_mapping_Hz if key in dict_mapping]
            ax2[i].hist(combined_mapping_Hz[key], bins=nbins)
            ax2[i].set_title(str(key) + ' ' + 'Hz')
            i = i +1
        
        list_dict_mapping_km = df['mapping_km'].tolist()
        combined_mapping_km = {0:[],100:[],200:[]}
        for key in combined_mapping_km:
            combined_mapping_km[key] = [dict_mapping[key] for dict_mapping in list_dict_mapping_km if key in dict_mapping]
            ax2[i].hist(combined_mapping_km[key], bins=nbins//50)
            ax2[i].set_title(str(key) + ' ' + 'km')
            i = i +1
            

        
def grid_default_values(regex_subdir, regex_images,min_subset=10):
    
    """Obtain default values to use for the grid, 
    separating based on whether the metadata is located on the left or bottom
    
    :param regex_subdir: regular expression to extract subdirectory ex: 'E:/master/R*/[0-9]*/'
    :type regex_img: str
    :param regex_img: regular expression to extract images ex: '*.png'
    :type regex_img: str
    :param min_subset: minimum number of items extracted to be considered ,defaults to 10
    :type min_subset: int, optional
    :returns: df_summary_bottom,df_summary_left i.e. Dataframe summarizing the grid values for metadata located on the left,Dataframe summarizing the grid values for metadata located on the bottom
    :rtype: class: `pandas.core.frame.DataFrame`, class: `pandas.core.frame.DataFrame`
    .. todo:: enable pickled values
    """
    # All the subdirectory i.e. R014207948/1743-9/
    list_all_subdir = glob.glob(regex_subdir)
    print(list_all_subdir)
    
    df_summary_bottom = pd.DataFrame(columns=['row_peaks','col_peaks','mapping_Hz', 'mapping_km'] )
    df_summary_left = pd.DataFrame(columns=['row_peaks','col_peaks','mapping_Hz', 'mapping_km'] )
    
    for subdir_name in list_all_subdir:
        print(subdir_name)
        try:
            df_img,_,_ =segment_images(subdir_name, regex_images)
            for metatype in ['left','bottom']:
                df_img_subset = df_img[df_img['metadata_type']==metatype]
                if len(df_img_subset.index) > min_subset + 1:
                        stack_subset = all_stack(df_img_subset)
                        col_peaks,row_peaks,mapping_Hz, mapping_km = get_grid_mappings(stack_subset,use_defaults=False)
                        to_apend = pd.DataFrame({'row_peaks':[row_peaks],'col_peaks':[col_peaks],'mapping_Hz':[mapping_Hz],'mapping_km':[mapping_km]} )
                        if metatype == 'left':
                            df_summary_left = df_summary_left.append(to_apend)
                        elif metatype == 'bottom':
                            df_summary_bottom = df_summary_left.append(to_apend)
         
        except Exception:
            traceback.print_exc()
            print(subdir_name )
            
    df_summary_bottom.name = 'bottom'
    df_summary_left.name = 'left'
                    
    return df_summary_bottom,df_summary_left

if __name__ == '__main__':
    df_summary_bottom1,df_summary_left1 = grid_default_values(regex_subdir='L:/DATA/ISIS/raw_upload_20230421/R*/B*/', regex_images='*.png')
    df_merge1 = df_summary_bottom1.append(df_summary_left1)
    df_merge1.name = 'merged'
    #plot_hist_peaks_grids(df_summary_bottom1,df_summary_left1,df_merge1)
