# Ashley Ferreira, CSA, December 2023

# version of scan_error_detection_multi_instance_ISIS.py
# that just focuses on cropped-too-soon errors

# imports
import sys
import os
from optparse import OptionParser
parser = OptionParser()
        
parser.add_option('-u', '--username', dest='username', 
        default='aferreira', type='str', 
        help='CSA network username, default=%default.')
      
parser.add_option('-s', '--save', dest='saveDir', 
        default='L:/DATA/ISIS/cropped_too_soon_detection/', type='str', 
        help='Path to directory where results output file should be saved, default=%default. All instances should output to the same path.')

parser.add_option('-f', '--filename', dest='filename', 
        default='cropped_too_soon_results.csv', type='str', 
        help='Name of file to output results to in the path $SAVEDIR$, default=%default. This document is also used to track which subdirectories have \
              already been processed so to avoid different instances duplicating efforts all instances should be outputting to the same file.')

(options, args) = parser.parse_args()

# replace this with your own library path for --user pip installs (if applicable)
sys.path.append(f'C:/Users/{options.username}/AppData/Roaming/Python/Python38/Scripts')

# more imports
import gc
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import string 

#imports - pixel reading (Jeysh 10/31/23)
from PIL import Image 
import cv2
import shutil

# set paths
batchDir = 'L:/DATA/ISIS/ISIS_101300030772/' # need to run for more newly-uploaded data too
saveDir = options.saveDir 
outFile = saveDir + options.filename
print('This program will be saving to results file location:', outFile)

# make the directory to save into  
# if it doesn't already exist
if not(os.path.exists(saveDir)):
    os.makedirs(saveDir)

append2outFile = True # change to false if starting new file, have not tested in long time
saveImages = False

def read_image(image_path, modified_name, saveDir=saveDir):
    try: 
        image = cv2.imread(image_path)
        # extract height and width of image in pixels 
        height, width = image.shape[0], image.shape[1]
        
        if height/width > 1:
            shutil.copy(image_path, saveDir + modified_name) 

        return height, width
    
    except:
        return -1, -1 # error code

def read_all_directories(outFile=outFile, append2outFile=append2outFile, batchDir=batchDir):
    # initialize lists to save values to in loop
    directories, subdirs, images = [], [], []
    heights, widths = [], []
    user_lst, datetime_lst = [], []

    # assume file has not yet been written to and needs header
    header = True

    # set column types for results dataframe 
    types = {'Directory': 'str', 'Subdirectory': 'str', 'filename': 'str', 
             'height': 'float32', 'width': 'float32', 'user': 'str', 'datetime':'str'}
    
    # loop over all directories in the batch 2 raw data directory
    raw_contents = os.listdir(batchDir) # random shuffle applied
    random.shuffle(raw_contents) # random shuffle applied
    for directory in raw_contents:

        # loop over all subdirectories within the directory
        directory_contents = os.listdir(batchDir + directory) 
        random.shuffle(directory_contents) # random shuffle applied
        for subdir in directory_contents:

            # in this approach we hope that no same subdirectory is found by two instances
            # at similar times but we check before saving to avoid writting duplicates
            # --> alternatively can sample from an "unprocessed" log
            print('###############################\n###############################')
            print('###############################\n###############################')
            print('dir:', directory, '\nsubdir:', subdir)
            print('###############################\n###############################')
            print('###############################\n###############################')

            # if path exists we want to see what subdirs are processed
            # this far and so that is what we check below
            if os.path.exists(outFile):

                # this is just a trick to make sure file isn't in use
                read_safe = False
                while read_safe == False:
                    try:
                        os.rename(outFile, outFile)

                        # get a set of already processed dirs and subdirs
                        df_processed_results = pd.read_csv(outFile, dtype=types)
                        subdir_id_lst = set(str(df_processed_results['Directory']) + ' ' + str(df_processed_results['Subdirectory']))

                        # clear memory of stuff we don't need
                        del df_processed_results

                        read_safe = True 

                    except (OSError, PermissionError) as e:
                        print(e)
                        print(outFile, 'currently being used, pausing for 30 seconds before another attempt')
                        time.sleep(30)

                # check that this specific subdir has already been processed
                if str(directory + ' ' + subdir) in subdir_id_lst:
                    print('this subdirectory has already been processed, moving on to the next one...')
                    proceed = False # the continue statement was not working as expected
            
                else:
                    print('this subdirectory is not already processed, beginning processing...')
                    proceed = True
            
            else: # TEMP
                print(f'error: outFile does not exist ({outFile}), did not save!!')
                proceed = False

            if proceed == True:
                # loop over all images in the subdirectory
                subdir_contents = os.listdir(batchDir + directory + '/' + subdir) 
                for image in subdir_contents:

                    # save full path of image
                    image_path = batchDir + directory + '/' + subdir + '/' + image

                    # save id of image
                    directories.append(directory)
                    subdirs.append(subdir)
                    images.append(image)
                    
                    modified_name = f'{directory}_{subdir}_{image}'

                    h, w = read_image(image_path, modified_name)

                    # save values
                    heights.append(h)
                    widths.append(w)
                    user_lst.append(options.username)
                    datetime_lst.append(datetime.datetime.now())


                #### SAVE RESULTS AFTER PROCESSING EACH SUBDIR ####                
                # initialize dataframe and save results to csv
                # (redoing this each iteration to not loose information)
                df_mapping_results = pd.DataFrame()
                df_mapping_results['Directory'] = directories
                df_mapping_results['Subdirectory'] = subdirs
                df_mapping_results['filename'] = images
                df_mapping_results['height'] = heights
                df_mapping_results['width'] = widths 
                df_mapping_results['user'] = user_lst
                df_mapping_results['datetime'] = datetime_lst

                # mode = 'a' means it will append to existing data within the file
                if append2outFile == True:
                    mode = 'a' 

                    # wipe lists now that they have been saved
                    directories, subdirs, images = [], [], []
                    heights, widths = [], []
                    user_lst, datetime_lst = [], []
                    
                else: # append2outFile = False should not be used for multi-instance
                    # this overwrites existing file
                    mode = 'w' # check exits or not instead?
                    header = True

                print('subdirectory', subdir, 'processed, attempting to save data...')
                save = True
                print(os.path.exists(outFile))
                if os.path.exists(outFile):
                    write_safe = False
                    while write_safe == False:
                        print('***')
                        try:
                            os.rename(outFile, outFile)

                            # get a set of already processed dirs and subdirs
                            # WHAT HAPPENS WHEN EMPTY? START IT YOURSELF NOW
                            df_processed_results = pd.read_csv(outFile, dtype=types)
                            subdir_id_lst = set(str(df_processed_results['Directory']) + ' ' + str(df_processed_results['Subdirectory']))

                            # check that this specific subdir has already been processed
                            if str(directory + ' ' + subdir) in subdir_id_lst:
                                print('thanks for all your hard work but you got unlucky and another instance processed this before this one\nNOT SAVING RESULTS')
                                # note: we should check for duplicates in the analysis just incase + check all data has been saved
                                save = False
                            
                            if save == True:
                                # check if there is already data in the output file 
                                # (this may create duplicate headers if instances finish 
                                # processing their first subdir too close together)
                                if header == True and os.path.exists(outFile) and os.path.getsize(outFile)!=0:
                                    header = False 

                                df_mapping_results.to_csv(outFile, mode=mode, index=False, header=header)
                                del df_mapping_results
                                del df_processed_results
                                print(f'data sucessfully saved to {outFile}')

                            write_safe = True 

                        except (OSError, PermissionError) as e:
                            print(e)
                            print(outFile, 'currently being used, pausing for 1 minute before another attempt')
                            time.sleep(60)

                collected = gc.collect()
                print("Garbage collector: collected",
                        "%d objects." % collected)


if __name__ == '__main__':
    read_all_directories()