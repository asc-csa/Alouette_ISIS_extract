
#Jeyshinee Pyneeandee - Nov 2023 - Flagging Over Exposed ISIS Ionograms

#imports 
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from skimage.color import rgb2gray
from skimage.exposure import histogram
from skimage.util import img_as_ubyte
import os, csv
import random
import pandas as pd
import time 
import gc

#Path to save results 
outFile ='L:/DATA/ISIS/OverExposure/Overexposed_and_cropped_too_soon_raw_upload.csv'
#ISIS Ionograms Directory
batchDir = 'L:/DATA/ISIS/cropped_too_soon_detection_batch3/cropped_too_soon_results_flagged.csv'

print('This program will be saving to results file location:', outFile)

def flag_overexposed(image_path, plotting_hist = False):
    '''
    Definition:
        flag_exposed takes in one ionogram and saves it if it is overexposed
    
    Parameters:
        image_path (str) : Path to image

        plotting_hist (bool) : Default= False. If true, display histogram plot 

    Return:
        saves over_exposed image 
    
    '''
    try:
        #Get frequency and bins for histogram
        path = image_path
        img = imageio.imread(path)
        image_intensity = img_as_ubyte(rgb2gray(img))
        freq, bins = histogram(image_intensity)
        width, height = img.shape[0], img.shape[1]
        total_pixels = width*height
       
       #Plot histogram, if true
        if plotting_hist:
            plt.step(bins, freq*1.0/freq.sum())
            plt.xlabel('intensity value')
            plt.ylabel('Fraction of pixels')
            plt.show()
        
        #Get histogram integral values for pixels >= 230
        integral_255 = 0
        for i in range(230,len(bins)-1):
            bin_width = bins[i+1] - bins[i]

            # Sum over number in each bin and multiply by bin width
            integral_255 = integral_255 + (bin_width * sum(freq[i:i+1]))
        proportion = integral_255/total_pixels
        #Flag ionogram if prop 0.11
        if proportion > 0.11:
            return proportion, True 
        else:
            return proportion, False

    except Exception as e:
        print("Error -", e)

def read_all_directories(outFile=outFile,batchDir=batchDir):

    overexposed = []
    directories, subdirs, images = [], [], []
    row_number = 0
    #opening csv file containing images cropped too soon
    file = open(batchDir)    
    heading = next(file)
    reader = csv.reader(file)
    for row in reader:
        image_path = 'L:/DATA/ISIS/raw_upload_20230421/' + row[0] + '/' + row[1]  + '/' + row[2]
        print(image_path)
        directories.append(row[0])
        subdirs.append(row[1])
        images.append(row[2])

        prop_val, bool_check = flag_overexposed(image_path)
        print("row number:", str(row_number))
        row_number = row_number + 1
        
        if bool_check:
            overexposed.append("True")
        else:
            overexposed.append("False")
                
    df_mapping_results = pd.DataFrame()
    df_mapping_results['Directory'] = directories
    df_mapping_results['Subdirectory'] = subdirs
    df_mapping_results['filename'] = images
    df_mapping_results['OverExposed'] = overexposed
    df_mapping_results.to_csv(outFile, mode='a', index=False)
    
if __name__ == '__main__':
    read_all_directories()

#Get counts     
# test_batch_2 = pd.read_csv('L:/DATA/ISIS/OverExposure/Overexposed_and_cropped_too_soon_raw_upload.csv')
# len(test_batch_2)

# file = open('L:/DATA/ISIS/OverExposure/Overexposed_and_cropped_too_soon_raw_upload.csv')
# heading = next(file)
# reader = csv.reader(file)
# count = 0
# for row in reader:
#     if row[0] == "True":
#         count += 1
# print(count)
# file.close()


# #stats
# count_overexposed_batch2 = 6327
# total_cropped_batch2 = 35963

# count_overexposed_batch2_addition = 73
# total_cropped_batch2_addition = 8

# count_overexposed_raw_upload = 5132
# total_cropped_raw_upload = 11549

# count_overexposed_batch1 = 14694
# total_cropped = 57091

