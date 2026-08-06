# -*- coding: utf-8 -*-
"""
Code to obtain numbers about the dataset, such as the total number of raw images
and the number of ionograms with metadata at the bottom vs number of ionograms with metadata on the left


"""
# Library imports
import glob

from segment_images_in_subdir import segment_images


def total_number_raw_images(directory):
    '''Returns all the raw images in a directory as well as the number of raw images
    
    :param directory: regular expression to isolate all the raw images in a directory as well as the number of raw images
    :type directory: string
    :returns: all_img, len(all_img ) i.e. list of all the raw images in a directory, length of all_img
    :rtype: list, int
    

    '''
    all_img = glob.glob(directory)
    
    return all_img, len(all_img )
    

def number_bottomside_vs_leftside(regex_subdir,regex_images):
    '''Returns the number of ionograms with metadata at the bottom vs number of ionograms with metadata on the left
    
    :param regex_subdir: regular expression to extract subdirectory ex: 'E:/master/R*/[0-9]*/'
    :type regex_img: str
    :param regex_img: regular expression to extract images ex:'E:/master/R*/[0-9]*/'
    :type regex_img: str
    :returns: total_bottom,total_left i.e. Number of ionograms with metadata at the bottom, Number of ionograms with metadata at the left
    :rtype: int, int
        
    '''
    # All the subdirectory i.e. R014207948/1743-9/
    list_all_subdir = glob.glob(regex_subdir)
    
    total_bottom = 0
    total_left = 0
    
    for sample_subdir in list_all_subdir:
        df_img,_,_ =segment_images(sample_subdir, regex_images)
        n_left = len(df_img[df_img['metadata_type'] == 'left'].index)
        n_bottom = len(df_img[df_img['metadata_type'] == 'bottom'].index)
        total_bottom = n_bottom + total_bottom
        total_left = total_left + n_left 
        
#    for i,pkl_path in enumerate(all_pkl):
#        df= pd.read_pickle(pkl_path)
#        type_data = np.asarray(df['trimmed_metadata'].map(lambda trimmed: 1 if trimmed.shape[0] < trimmed.shape[1] else 0))
#        
#        total_bottom = total_bottom + sum(type_data)
#        total_left = total_left + sum(1-type_data)

            
    return total_bottom,total_left

if __name__ == '__main__':
    
    # Number of images in each hard disk 
    all_img,len_all_img  =  total_number_raw_images(regex_img='E:/master/R*/[0-9]*/*.png' )
    all_img2, len_all_img2 = total_number_raw_images(regex_img='G:/AlouetteData/Alouette Data/R*/[0-9]*/*[0-9].png' )
    total_bottom,total_left = number_bottomside_vs_leftside(regex_subdir='E:/master/R*/[0-9]*/', regex_images='*.png')
    total_bottom2,total_left2 = number_bottomside_vs_leftside(regex_subdir='G:/AlouetteData/Alouette Data/R*/', regex_images='G:/AlouetteData/Alouette Data/R*/[0-9]*/*[0-9].png')
    
    
    
    

