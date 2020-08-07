# -*- coding: utf-8 -*-
"""
Visualize the output of code to translate leftside metadata
"""
# Library imports
import random
import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt
from textwrap import wrap


from translate_leftside_metadata import get_leftside_metadata


sys.path.append('../')
from image_segmentation.segment_images_in_subdir import segment_images
from helper_functions import generate_random_subdirectory

LABELS_DICT = ['dict_cat_dot','dict_num_dot','dict_cat_digit','dict_num_digit']

def test_leftside_metadata(sample_subdirectory,regex_images):
    """Visualize the output of translate_leftside_metadata.get_leftside_metadata on a random image from a subdirectory
    
    :param sample_subdirectory: sample subdir to test get_leftside_metadata
    :type sample_subdirectory: str
    :param regex_images: regular expression to extract images ex: '*.png'
    :type regex_images: str
    """
    

    no_leftside = True
    while no_leftside == True:
        # Segment images in the subdirectory
        df_img,_,_ =segment_images(sample_subdirectory, regex_images)
        
        # Type of metadata ('left' or 'bottom') in the subdirectory
        list_metadata_type = df_img['metadata_type'].tolist()
        type_metadata_subdir = max(set(list_metadata_type), key=list_metadata_type.count)
        
        # Get the metadata values
        if type_metadata_subdir == 'left':
            df_img_left = df_img[df_img['metadata_type']== 'left']
            df_img_left, df_loss_meta,dict_mapping,dict_hist= get_leftside_metadata(df_img_left,sample_subdir)
            
            # Visualize a random row of the dataframe output of df_img
            n_row_df = len(df_img_left.index)
            ix_row = random.randint(0,n_row_df -1)
            row = df_img_left.iloc[ix_row,:]
            
            fig,axes = plt.subplots(nrows=3,ncols=2)
            fig.suptitle(row['file_name'])
            ax = axes.ravel()
            ax[0].imshow(row['raw'],'gray')
            dict_metadata = row['dict_metadata']
            
            if row['is_dot']:
                type_row_meta = ' dot '
                labels= ['day_1','day_2','day_3','hour_1','hour_2','minute_1','minute_2','second_1', 'second_2','station_code']
                x_peaks = dict_mapping['dict_cat_dot']
                y_peaks = dict_mapping['dict_num_dot']
            else:
                type_row_meta = ' num '
                labels = ['satellite_number','year','day_1','day_2','day_3','hour_1','hour_2','minute_1','minute_2','second_1', 'second_2', 'station_number_1','station_number_2']
                x_peaks = dict_mapping['dict_cat_digit']
                y_peaks = dict_mapping['dict_num_digit']
            
            title = row['file_name']+ type_row_meta+ '\n'+ str([label+'='+ str(sum(dict_metadata[ label])) if label in dict_metadata.keys() 
                                            else label+'=0' for label in labels])

            h,w = row['dilated_metadata'].shape
            backtorgb = cv2.cvtColor(row['dilated_metadata'],cv2.COLOR_GRAY2RGB)


            for i in x_peaks.keys():
                cv2.line( backtorgb , (i, 0), (i,h), [0,0,255],2)
            for i in y_peaks.keys():
                cv2.line( backtorgb , (0, i), (w,i),[0,0,255],2)
                
            wrapped_title = ("\n".join(wrap(title, 60)))
            ax[1].imshow(backtorgb )
            ax[1].set_title(wrapped_title)
            plt.tight_layout()
            
            
            ax[2].set_title('x_dot: peaks used to create dict_cat_dot ')
            ax[3].set_title('y_dot: peaks used to create dict_num_dot ')
            ax[4].set_title('x_num: peaks used to create dict_cat_digit')
            ax[5].set_title('y_num: peaks used to create dict_num_digit')
            
        
            for i,dict_type in enumerate(LABELS_DICT):
                try:
                    idx_peaks,bin_edges,counts = dict_hist[dict_type]
                    bin_centers = (0.5*(bin_edges[1:] + bin_edges[:-1]))
                    peaks = bin_edges[np.array(idx_peaks)]
                    ax[i+2].plot(bin_centers,counts)
                    ax[i+2].plot(peaks,counts[idx_peaks], "x")
                except:
                    print('no ' + dict_type)
                    continue
                    
            no_leftside = False

if __name__ == '__main__':
    sample_subdir = generate_random_subdirectory(regex_subdirectory='G:/AlouetteData/Alouette Data/R*/[0-9]*[0-9]/')
    test_leftside_metadata(sample_subdir, regex_images='Image*[0-9].png')
    
    sample_subdir = generate_random_subdirectory(regex_subdirectory='E:/master/R*/[0-9]*/')
    test_leftside_metadata(sample_subdir , regex_images='*.png')



            
           
                
