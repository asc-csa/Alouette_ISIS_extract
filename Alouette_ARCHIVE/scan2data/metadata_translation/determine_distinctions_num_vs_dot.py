# -*- coding: utf-8 -*-
"""
Code to determine feature values to distinguish metadata containing dots to those containing numbers
- used for leftside_metadata_grid_mapping
"""

# Library imports
import sys
import glob


import cv2
import pandas as pd
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

sys.path.append('../')
from image_segmentation.extract_ionogram_from_scan import extract_ionogram
from image_segmentation.extract_metadata_from_scan import extract_metadata
from image_segmentation.trim_raw_metadata import trimming_metadata


def extract_meta_info(dilated_meta,
                      min_num_pixels=50, max_number_pixels=1000):
    """Extract key informational values about the metadata in dilated_meta
    
    :param dilated_meta: trimmed metadata (output of image_segmentation.leftside_metadata_trimming) after a rotation and dilation morphological transformation (see translate_leftside_metadata.extract_leftside_metadata )
    :type dilated_meta: class: `numpy.ndarray`
    :param min_num_pixels: minimum number of pixels to be considered metadata dot/num, defaults to 50
    :type min_num_pixels: int, optional
    :param max_number_pixels: maximum number of pixels to be considered metadata dot/num, defaults to 1000
    :type max_number_pixels: int, optional
    :returns: mean_area ,median_area ,h,w,h*w: mean area of the metadata dots/numbers, median area of the metadata dots/numbers, height of the trimmed rectangle containing metadata, width of the trimmed rectangle containing metadata, height*width
    :rtype: class: float, float, float, float, float
    :raises Exception: returns np.nan,np.nan,h,w,h*w if there is an error
    
    """
    try:
        h,w = dilated_meta.shape
        _, _, stats, centroids	=	cv2.connectedComponentsWithStats(dilated_meta)
        area_centroids = stats[:,-1]

        area_centroids =area_centroids[np.logical_and(area_centroids > min_num_pixels, area_centroids < max_number_pixels)]
        mean_area = np.mean(area_centroids)
        median_area = np.median(area_centroids)

        return mean_area ,median_area ,h,w,h*w
    except Exception:
        return np.nan,np.nan,h,w,h*w


def generate_sample(regex_all_raw_img, sample_size):
    """Dataframe sample to determine features to distinguish metadata containing dots to those containing numbers
    
    :param regex_all_raw_img: regular expression to extract all raw images ex: 'E:/master/R*/[0-9]*/*.png' 
    :type regex_all_raw_img: str
    :param sample_size: number of raw images to generate dataframe sample and eventually label
    :type sample_size: int
    :returns: df_sample i.e. Dataframe of characteristics describing sample_size random metadata
    :rtype: class: `pandas.core.frame.DataFrame`
    """
    list_all_raw_img = np.array(glob.glob(regex_all_raw_img))
    sample = list_all_raw_img[np.random.choice(len(list_all_raw_img),sample_size,replace=False)]
    df_sample = pd.DataFrame(data = {'file_name': sample})    
    df_sample['raw'] = df_sample['file_name'].map(lambda file_name: cv2.imread(file_name,0))
    
    
    df_sample['limits'],df_sample['ionogram'] = zip(*df_sample['raw'].map(lambda raw_img: extract_ionogram(raw_img)))
    loss_ion_extraction = df_sample.isna().any(axis=1)
    df_sample = df_sample[~loss_ion_extraction]
    
    df_sample['metadata_type'],df_sample['raw_metadata'] = zip(*df_sample.apply(lambda row: extract_metadata(row['raw'], row['limits']),1))
    loss_metadata = np.any([df_sample['metadata_type'] == 'right',df_sample['metadata_type']=='top'],axis=0)
    df_sample = df_sample[~loss_metadata]
    
    df_sample['trimmed_metadata'] = df_sample.apply(lambda row: trimming_metadata(row['raw_metadata'],row['metadata_type']) , 1)
    loss_trim = df_sample.isna().any(axis=1)
    df_sample = df_sample[~loss_trim]
    
    df_sample['rotated_metadata'] = df_sample['trimmed_metadata'].map(lambda trimmed_meta: np.rot90(trimmed_meta,-1))
        
    kernel_dilation = np.ones((1,1),np.uint8)
    df_sample['dilated_metadata'] = df_sample['rotated_metadata'].map(lambda rotated_meta: cv2.dilate(rotated_meta,kernel_dilation))
    
    df_sample['mean_area'],df_sample['median_area'],df_sample['h'],df_sample['w'],df_sample['h*w'] = zip(*df_sample['dilated_metadata'].map(lambda dilated: extract_meta_info(dilated)))

    return df_sample

if __name__ == '__main__':
    # Try for 100 random images
    df = generate_sample('G:/AlouetteData/Alouette Data/R*/[0-9]*[0-9]/Image*[0-9].png', sample_size = 100)
   
    # Generate pdf file for labelling
    with PdfPages('test_dots_vs_num.pdf') as pdf:
        for i,row in df.iterrows():
            fig,axes = plt.subplots(2)
            ax = axes.ravel()
            fig.suptitle(row['file_name'] )
            ax[0].imshow(row['raw'], 'gray')
            ax[1].imshow(row['trimmed_metadata'], 'gray')
            plt.close('all')
            pdf.savefig(fig,dpi=300)

    # Manually enter labels for each picture
    df['labels']=labels

   # Obtain features
    df_test = df[['file_name','mean_area', 'median_area', 'h', 'w', 'h*w','labels']]
    df_test_dot = df_test[df_test['labels'] == 0]
    df_test_dot = df_test_dot[['mean_area', 'median_area', 'h', 'w', 'h*w']].agg(['min','max','mean'])
    df_test_num = df_test[df_test['labels'] == 1]
    df_test_num = df_test_num[['mean_area', 'median_area', 'h', 'w', 'h*w']].agg(['min','max','mean'])