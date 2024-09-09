# Ashley Ferreira, CSA, June 2023

# notes:
# - this is  a version of scan_error_detection which supports multiple instances and users
# - to kill this program: go ctrl+c
# - reflects the new (more correct) naming convention that roll = directory
# ^ other code Ashley has created does not yet reflect this update

# run these pip commands in anaconda prompt to download non-standard libraries (may need to add --user)
# >pip install tensorflow (or for GPU use: >pip install tensorflow==2.10)
# >pip install keras_ocr

# imports
import sys
import os
from optparse import OptionParser
parser = OptionParser()

### THE ONLY REQUIRED ARGUMENT IS USERNAME, THEN IF YOU WAN TO USE GPU YOU ALSO ####
##### NEED TO CHANGE -d TO GPU and -e TO YOUR TENSORFLOW 2.10 ENVIRONMENT NAME #####
######## ALL INSTANCES SHOULD OUTPUT TO THE SAME FILE TO AVOID DUPLICATIONS ########

parser.add_option('-d', '--device', dest='device', 
        default='CPU', type='str', 
        help='Device to run TensorFlow on. Can only be "GPU" or "CPU", default=%default.')
        
parser.add_option('-u', '--username', dest='username', 
        default='aferreira', type='str', 
        help='CSA network username, default=%default.')
        
parser.add_option('-e', '--env_name', dest='env_name', 
        default='tf210', type='str', 
        help='Path of TensorFlow 2.10.* environment to go within $ENV$ in u:/temp/$USERNAME$/python/envs/$ENV$/lib/site-packages (if not just using base env), default=%default.')
        
parser.add_option('-s', '--save', dest='saveDir', 
        default='L:/DATA/Alouette_I/BATCH_II_scan_error_detection_Run1/', type='str', 
        help='Path to directory where results output file should be saved, default=%default. All instances should output to the same path.')

parser.add_option('-f', '--filename', dest='filename', 
        default='results.csv', type='str', 
        help='Name of file to output results to in the path $SAVEDIR$, default=%default. This document is also used to track which subdirectories have \
              already been processed so to avoid different instances duplicating efforts all instances should be outputting to the same file.')

(options, args) = parser.parse_args()

# replace this with your own library path for --user pip installs (if applicable)
sys.path.append('C:/Users/' + options.username + '/AppData/Roaming/Python/Python38/Scripts')

if options.device == 'CPU':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
if options.device == 'GPU':
    sys.path.insert(0, 'U:/temp/' + options.username + '/python/envs/' + options.env_name + '/lib/site-packages/')

# more imports
import gc
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import keras_ocr

print('tensorflow version (should be 2.10.* for GPU compatibility)', tf.__version__)
if len(tf.config.list_physical_devices('GPU')) != 0: 
    print('GPU in use for tensorflor')
else:
    print('CPU in use for tensorflow')

pipeline = keras_ocr.pipeline.Pipeline()

# set paths
batchDir = 'L:/DATA/Alouette_I/BATCH_II_raw/'
saveDir = options.saveDir 
outFile = saveDir + options.filename
print('This program will be saving to results file location:', outFile)

# make the directory to save into  
# if it doesn't already exist
if not(os.path.exists(saveDir)):
    os.makedirs(saveDir)

# set default saving settings
# (not sure if code works anymore if these are changed)
append2outFile = True 
saveImages = False


def read_image(image_path, plotting=False, just_digits=False):
    '''
    This function reads in one image a time and outputs the height 
    and width along with the estimated digit count of the metadata.

    Parameters:

        image_path (str): path to the image

        plotting (bool, optional): True for a verbose display mode to
                                   illustrate the analysis in detail, 
                                   False otherwise

        just_digits (bool, optional): if True only count characters that are 
                                      integers, False to count any characters

    Returns:

        digit_count (int): estimated number of integers in the ionogram
                          metadata (right now, only looks for numbers along
                          bottom 20% of the image, usually only 15 expected)
                          ~~~~ done for cropped portion of image ~~~~
    
        height (int): number of pixels along y-axis of original image
        
        width (int): number of pixels along x-axis of origional image

        says_isis (bool): True if 'isis' independant of capitalization is 
                          present within the detected text, False otherwise
                          ~~~~ done for origional image ~~~~
    '''
    try: 

        # read in image using keras_ocr
        image = keras_ocr.tools.read(image_path) 

        # extract height and width of image in pixels 
        height, width = image.shape[0], image.shape[1]

        # cut image to just include bottom 20% of pixels
        cropped_height = height-height//5

        # create predictions for location and value of characters
        # on the cropped image, will output (word, box) tuples
        prediction = pipeline.recognize([image])[0]

        # if no characters are found move on
        if prediction == [[]]:
            digit_count = 0

        # if characters are found look at the predictions
        else:
            if plotting == True:
                # plot the predictied box and tuples
                keras_ocr.tools.drawAnnotations(image=image, predictions=prediction)
                plt.show()

            # loop over predicted (word, box) tuples and count number of digit characters
            digit_count = 0
            says_isis = False
            for p in prediction:

                # select word and box part of the tuple
                value, box = p[0], p[1]

                # check for 'isis' of any capitalization in image
                if 'isis' in value.lower(): # may want to check 1, I, 5 variations on this to detect like 15iS
                    says_isis = True
                    print('found potential ISIS text')
                
                # if word is composed of just integers then 
                # count how many and incriment digit_count
                if just_digits == False or (just_digits == True and value.isdigit()):
                    # check that box is within the cropped height
                    in_bounds = True
                    for b in box:
                        if b[1] < cropped_height:
                            in_bounds = False
                            break
                            
                    if in_bounds:
                        digit_count += len(value)

        print('digits count:', digit_count)

    except Exception as e:
        print('ERR:', e)
        digit_count, height, width, says_isis = -1, -1, -1, 'ERR'

    return digit_count, height, width, says_isis


def read_all_directories(outFile=outFile, append2outFile=True, batchDir=batchDir, plotting=False):
    '''
    This function loops over all images nested within batchDir
    and saves the outputs from read_image() to a CSV file.

    Parameters:

        outFile (str, optional): path to CSV file where results from this 
                                function can be stored 

        append2outFile (bool, optional): if True will append to data in outFile 
                                        (if any exists), otherwise overwrites

        batchDir (str, optional): path to directory of entire batch 
                                    of ionogram scan images to analyze

        plotting (bool, optional): just passes directly to read_image()

    Returns:

        None

    '''
    # initialize lists to save values to in loop
    directories, subdirs, images = [], [], []
    heights, widths, digit_counts = [], [], []
    says_isis_lst, user_lst = [], []

    # assume file has not yet been written to and needs header
    header = True

    # set column types for results dataframe 
    types = {'Directory': 'str', 'Subdirectory': 'str', 'filename': 'str', 'digit_count': 'int', \
             'height': 'float32', 'width': 'float32', 'says_isis': 'str', 'user': 'str'}
    
    # loop over all directories in the batch 2 raw data directory
    raw_contents = os.listdir(batchDir) # random suffle appled
    random.shuffle(raw_contents) # random suffle applied
    for directory in raw_contents:

        # loop over all subdirectories within the directory
        directory_contents = os.listdir(batchDir + directory) 
        random.shuffle(directory_contents) # random shuffle applied
        for subdir in directory_contents:

            # in this approach we hope that no same subdirectory is found by two instances
            # at similar times but we check before saving to avoid writting duplicates
            # --> alternatively can sample from an "unprocessed" log

            print('###############################')
            print('dir:', directory, '\nsubdir:', subdir)

            # if path exists we want to see what subdirs are processed
            # this far and so that is what we check below
            if os.path.exists(outFile):

                # this is just a trick to make sure file isnt in use
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
                    continue
            
                else:
                    print('this subdirectory is not already processed, beginning processing...')
            
            # loop over all images in the subdirectory
            subdir_contents = os.listdir(batchDir + directory + '/' + subdir) 
            for image in subdir_contents:

                # save full path of image
                image_path = batchDir + directory + '/' + subdir + '/' + image

                # save id of image
                directories.append(directory)
                subdirs.append(subdir)
                images.append(image)

                # send to read_image to get aspect ratio, digit count, and isis text
                num_of_digits, h, w, says_isis = read_image(image_path, plotting=plotting)

                # save values
                digit_counts.append(num_of_digits)
                heights.append(h)
                widths.append(w)
                says_isis_lst.append(says_isis)
                user_lst.append(options.username)


            #### SAVE RESULTS AFTER PROCESSING EACH SUBDIR ####                
            # initialize dataframe and save results to csv
            # (redoing this each interation to not loose information)
            df_mapping_results = pd.DataFrame()
            df_mapping_results['Directory'] = directories
            df_mapping_results['Subdirectory'] = subdirs
            df_mapping_results['filename'] = images
            df_mapping_results['digit_count'] = digit_counts
            df_mapping_results['height'] = heights
            df_mapping_results['width'] = widths
            df_mapping_results['says_isis'] = says_isis_lst
            df_mapping_results['user'] = user_lst

            # mode = 'a' means it will append to existing data within the file
            if append2outFile == True:
                mode = 'a' 

                # wipe lists now that they have been saved
                directories, subdirs, images = [], [], []
                heights, widths, digit_counts = [], [], []
                says_isis_lst, user_lst = [], []
                
            else: # append2outFile = False should not be used for multi-instance
                # this overwrites existing file
                mode = 'w'
                header = True

            print('subdirectory', subdir, 'processed, attempting to save data...')
            save = True
            if os.path.exists(outFile):
                write_safe = False
                while write_safe == False:
                    try:
                        os.rename(outFile, outFile)

                        # get a set of already processed dirs and subdirs
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
                            print('data sucessfully saved')

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
