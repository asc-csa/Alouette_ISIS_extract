# -*- coding: utf-8 -*-
"""

Visualize the entire dataset by plotting a subset of images of each subsubdirectory
in a large PDF to determine, among other things, which subdirectory requires flipping

"""

# Library imports
import random
import math

import cv2
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import glob

from extract_ionogram_from_scan import extract_ionogram

def subset_preview(regex_subdir,regex_img, output_file,
                   ionogram_preview=False,len_subset=12,n_rows_page = 4):
    """ Plotting a subset of images of each subsubdirectory in a large PDF 
    
    :param regex_subdir: regular expression to extract all the subdirectories ex: 'E:/master/R*/[0-9]*/'
    :type regex_img: str
    :param regex_img: regular expression to extract images ex: '*.png'
    :type regex_img: str
    :param output_file: path to place output pdf file
    :type output_file: str
    :param ionogram_preview: if only the ionogram should be extracted from the raw images, defaults to False
    :type ionogram_preview: bool, optional
    :param len_subset: number of images to randomly extract from a subdirectory, defaults to 12
    :type len_subset: int, optional
    :param n_rows_page: number of rows for each page, defaults to 4
    :type n_rows_page: int, optional
    """
    
    # List of all the subdirectories
    all_subdir =  glob.glob(regex_subdir)
    
    # Set up each pdf page 
    n_cols_page = int(math.ceil(len_subset/ n_rows_page))
    
    
    # Save random subset for each direcctory in a pdf
    with PdfPages(output_file) as pdf:
        for subdir in all_subdir:
            all_img_subdir = glob.glob(subdir+regex_img)
            fig,axes = plt.subplots(n_rows_page,n_cols_page)
            ax = axes.ravel()
            fig.suptitle(str(subdir))
            len_name_subdir = len(subdir)
            count = 0
            while count != len(ax):
                idx = random.randint(0,len(all_img_subdir)-1)
                img_path = all_img_subdir[idx]
                img = cv2.imread(img_path,0)
                if ionogram_preview:
                    _,ion = extract_ionogram(img)
                    if not type(ion) == float:
                        ax[count].imshow(ion, 'gray')
                        name_plot = img_path[len_name_subdir:]
                        ax[count].set_title(name_plot)
                        ax[count].axis('off') 
                        count = count + 1
  
                else:
                    ax[count].imshow(img, 'gray')
                
                    name_plot = img_path[len_name_subdir:]
                    ax[count].set_title(name_plot)
                    ax[count].axis('off') 
                    count = count + 1

            pdf.savefig(fig)
            plt.close('all')
if __name__ == '__main__':
    #subset_preview('E:/master/R*/[0-9]*/','*.png', 'subset_preview.pdf')
    subset_preview('G:/AlouetteData/Alouette Data/R*/[0-9]*[0-9]/','Image*[0-9].png', 'subset_preview_2.pdf',ionogram_preview=True)


    

