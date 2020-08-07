# -*- coding: utf-8 -*-
"""
Code to test out OCR (pytesseract) on the bottomside metadata
while plotting out the outputs in a pdf

"""


# Library imports
import random
import glob
import shutil
import sys

import cv2
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import pytesseract

sys.path.append('../')
from image_segmentation.extract_ionogram_from_scan import extract_ionogram
from image_segmentation.extract_metadata_from_scan import extract_metadata
from image_segmentation.trim_raw_metadata import trimming_metadata
def create_bottomside_subset(regex_img, len_subset,output_path):
    '''
    Create a subset of len_subset raw images with bottomside metadata at output_path
    
    :param regex_img: regular expression to extract all images ex: 'E:/master/R*/[0-9]*/*.png'
    :type regex_img: str
    :param len_subset: desired length of subset
    :type len_subset: int
    :param output_path: path of output
    :type output_path: str
    
    '''
    # All raw images
    all_img = glob.glob(regex_img)
    len_all_img = len(all_img)

    # Subset of raw images with metadata at the bottom
    count = 0
    count_threshold = len_subset
    list_subset = []
    
    while count < count_threshold:
        try:
            # Randomly pick an iamge
            idx = random.randint(0,len_all_img-1)
            test_file_path = all_img[idx]
            raw_img = cv2.imread(test_file_path,0)
            limits, _ = extract_ionogram(raw_img)
            type_meta,raw_meta= extract_metadata(raw_img, limits)
            
            # Add to subset if metadata at the bottom
            if type_meta == 'bottom':
                list_subset.append(test_file_path)
                count = count + 1
        except:
            print(test_file_path)
            continue
    
    for file_name in list_subset:
        shutil.copy(file_name,output_path)    
        


if __name__ == '__main__':
    regex_all_raw_images = 'E:/master/R*/[0-9]*/*.png'
    lenght_of_subset = 100
    output_dir_path = 'C:/temp/Alouette/meta_bottom'
    
    # Create susbet of raw images with metadata on the bottom
    create_bottomside_subset(regex_all_raw_images,lenght_of_subset,output_dir_path)
    
    # Plot the output of the OCR in a pdf
    name_pdf = 'test_meta_bottom.pdf'
    all_img_path = glob.glob(output_dir_path + '/*.png')
    with PdfPages(name_pdf) as pdf:
        df_loss = pd.DataFrame(columns = [ 'file_name','func_name', 'details'])
        for test_file_path in all_img_path:
            try: 
                raw_img = cv2.imread(test_file_path,0)
                limits, _ = extract_ionogram(raw_img)
                type_meta,raw_meta= extract_metadata(raw_img, limits)
                trimmed_meta = trimming_metadata(raw_meta,type_meta)
                
                # Plot running OCR on different versions of the metadata
                fig,axes = plt.subplots(5,2)
                ax = axes.ravel()
                fig.suptitle(test_file_path )
                ocr0 = pytesseract.image_to_string(raw_meta, lang='eng', config='--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789')
                ax[0].set_title("Meta_raw  \\"+str(ocr0))
                ax[0].imshow(raw_meta, 'gray')
                
                ocr1 = pytesseract.image_to_string(trimmed_meta, lang='eng', config='--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789')
                ax[1].set_title("Meta_trimmed  \\"+str(ocr1))
                ax[1].imshow(trimmed_meta, 'gray')
                
                # Enhance image thorugh histogram equalization
                equalized_meta = cv2.equalizeHist(trimmed_meta)
                
                ocr2 = pytesseract.image_to_string(equalized_meta, lang='eng',  config='--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789')
                ax[2].set_title("Meta_trimmed + equalized  \\"+str(ocr2))
                ax[2].imshow(equalized_meta, 'gray')
                
                threshold = 200
                _,binarized_meta = cv2.threshold(equalized_meta, threshold, 255, cv2.THRESH_BINARY)
                ocr3 = pytesseract.image_to_string(binarized_meta  , lang='eng', config='--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789')
                ax[3].set_title("Meta_trimmed + binarized  \\"+str(ocr3))
                ax[3].imshow(binarized_meta , 'gray')
                
                dict_parameter = {1:(1,(5,5)),2:(5,(5,5)),3:(20,(5,5)),4:(10,(3,3)),5:(10,(5,5)),6:(10,(10,10))}
                for i in range(4,10):
                    
                    clahe_meta = cv2.createCLAHE(trimmed_meta,*dict_parameter[i])
                    ocr = pytesseract.image_to_string(clahe_meta, lang='eng', config='--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789')
                    ax[i].set_title("Meta_trimmed + equalized"+str(i)+"  \\"+str(ocr))
                    ax[i].imshow(clahe_meta, 'gray')
                                
                plt.close('all')
                pdf.savefig(fig,dpi=300)
            except:
                print(test_file_path)
                continue