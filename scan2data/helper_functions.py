# -*- coding: utf-8 -*-
"""
Functions that are common for each submodule
"""
import glob
import random

def record_loss(df,function_name,subdir_location,
                columns_to_extract=['file_name'],loss_extraction=[]):
    """Generate a dataframe that records loss due to a self-imposed filter or a runtime programming error
    
    :param df: dataframe containing information of which image files did not pass a self-imposed filter or lead to runtime programming errors
    :type df: class: `pandas.core.frame.DataFrame`
    :param function_name: function or self-imposed filter leading to a loss
    :type function_name: str
    :param subdir_location: full path of the subdir
    :type subdir_location: str
    :param columns_to_extract: list of columns of df to extract, defaults to ['file_name']
    :type columns_to_extract: list, optional
    :param loss_extraction: whether a custom series is to be used to extract selected rows from the dataframe, defaults to []
    :type loss_extraction: class: `pandas.core.series.Series`, optional
    :returns: df_loss_extraction,loss_extraction i.e. dataframe containing file names leading to runtime errors or that do not pass pre-established filters (metadata size, ionogram size) as well as boolean series indicating which row of data to remove (==1)
    :rtype: (class: `pandas.core.frame.DataFrame`,class: `pandas.core.series.Series`)
    """   
    if len(loss_extraction) == 0:
        # function should return NA if there an error
        loss_extraction = df.isna().any(axis=1)
    # Record the files whose extraction was not successful
    df_loss_extraction = df[loss_extraction].copy()
    df_loss_extraction = df_loss_extraction[columns_to_extract]
    df_loss_extraction['func_name'] = function_name
    df_loss_extraction[ 'subdir_name'] = subdir_location
    
    return df_loss_extraction,loss_extraction


def generate_random_subdirectory(regex_subdirectory):
    """Extract random subdirectory
    
    :param regex_subdirectory: regular expression to extract subdirectory paths ex: 'E:/master/R*/[0-9]*/'
    :type regex_subdirectory: str
    :returns: sample_subdirectory: path of random subdirectory
    :rtype: str
    """
    # All the subdirectory i.e. ./R014207948/1743-9/
    list_all_subdirectory = glob.glob(regex_subdirectory)
    
    # Randomly pick a subdirectory
    return list_all_subdirectory[random.randint(0,len(list_all_subdirectory) - 1)]

def generate_random_image_from_subdirectory(subdirectory,regex_images):
    """Extract random raw image from a subdirectory
    
    :param subdirectory: path of subdirectory
    :type subdirectory: str
    :param regex_images: regular expression to extract images ex: '*.png'
    :type regex_images: str
    :returns: sample_subdirectory: path of random raw image in subdirectory
    :rtype: str
    """
    # All the images
    list_all_img = glob.glob(subdirectory+regex_images)

    # Randomly pick an image file
    return list_all_img[random.randint(0,len(list_all_img) - 1)]