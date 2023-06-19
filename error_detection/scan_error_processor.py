#Process subdirectories for possible scan errors

import sys
import pandas as pd
#import numpy as np
import os
from random import randrange
import time
from datetime import datetime
import gc

import warnings
warnings.filterwarnings('ignore')


#Set directories
dataDir = sys.argv[1]
rootDir = sys.argv[2]
resultDir = rootDir + '01_result/'
logDir = rootDir + '02_log/'

#Set parameters
env_name = sys.argv[3]
user_prefix = sys.argv[4]
instance = sys.argv[5]
user = user_prefix + instance #e.g: 'rnaidoo2'
stop_loop_threshold = 3000 #max while loops to prevent infinite loop


sys.path.insert(0, 'U:/temp/' + user_prefix + '/python/envs/' + env_name + '/lib/site-packages/')
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
print('tensorflow version (should be 2.10.* for GPU compatibility)', tf.__version__)
if len(tf.config.list_physical_devices('GPU')) != 0: 
    print('GPU in use for tensorflor')
else:
    print('CPU in use for tensorflow')

import keras_ocr
pipeline = keras_ocr.pipeline.Pipeline()



#Functions
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
                #plt.show()

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
        digit_count, height, width, says_isis = 'ERR', 'ERR', 'ERR', 'ERR'

    return digit_count, height, width, says_isis


def draw_random_subdir(subdir_ids_list, logDir):
    
    subdir_id = subdir_ids_list[randrange(len(subdir_ids_list))]
    subdir_id_parts = subdir_id.split('_')
    directory = subdir_id_parts[0]
    subdirectory = subdir_id_parts[1]
    
    #Check randomly-selected directory and subdirectory against the 'process_log'
    if os.path.exists(logDir + 'process_log.csv'):
        df_log = pd.read_csv(logDir + 'process_log.csv')
        df_search = df_log.loc[(df_log['Directory'] == directory) & (df_log['Subdirectory'] == subdirectory)]
        if len(df_search) > 0:
            print(directory + '/' + subdirectory + ' already processed!')
            return ''
        else:
            return directory, subdirectory
    else:
        return directory, subdirectory



#Process remaining subdirectories with while loop
stop_condition = False
stop_condition_counter = 0

while stop_condition == False:
    start = time.time()
    
    #Draw random, yet to be processed subdirectory, to process
    df_inventory = pd.read_csv(logDir + 'image_inventory.csv')
    subdir_ids_tot = df_inventory['subdir_id'].unique()
    if os.path.exists(logDir + 'scan_error_detect_log.csv'):
        df_log = pd.read_csv(logDir + 'scan_error_detect_log.csv')
        subdir_ids_proc = df_log['subdir_id'].unique()
    else:
        subdir_ids_proc = []
    subdir_ids_rem = list(set(subdir_ids_tot) - set(subdir_ids_proc))
    directory, subdirectory = draw_random_subdir(subdir_ids_list=subdir_ids_rem, logDir=logDir)
    subdir_path_end = directory + '/' + subdirectory + '/'
    
    #Process subdirectory
    print('')
    print('Processing ' + subdir_path_end + ' subdirectory...')
    print(str(len(subdir_ids_rem)) + ' subdirectories to go!')
    #img_fns = []
    df_result = pd.DataFrame()
    for file in os.listdir(dataDir + subdir_path_end):
        #img_fns.append(dataDir + subdir_path_end + file)
        image_path = dataDir + subdir_path_end + file
        num_of_digits, h, w, says_isis = read_image(image_path)
        row = pd.DataFrame({
            'Directory': directory,
            'Subdirectory': subdirectory,
            'filename': file,
            'digit_count': num_of_digits,
            'height': h,
            'width': w,
            'says_isis': says_isis
            }, index=[0])
        df_result = pd.concat([df_result, row])
    
    #Save:
    os.makedirs(resultDir + directory + '/', exist_ok=True)
    df_result.to_csv(resultDir + directory + '/' + 'result_scan_error_detect-' + directory + '_' + subdirectory + '.csv', index=False)
    
    end = time.time()
    t = end - start
    print('Processing time for subdirectory: ' + str(round(t/60, 1)) + ' min')
    print('Processing rate: ' + str(round(t/len(os.listdir(dataDir + subdir_path_end)), 2)) + ' s/img')
    print('')
    
    #Record performance
    df_log_ = pd.DataFrame({
        'Directory': directory,
        'Subdirectory': subdirectory,
        'Process_time': t,
        'Process_timestamp': datetime.fromtimestamp(end),
        'User': user,
        'subdir_id': directory + '_' + subdirectory
    }, index=[0])
    if os.path.exists(logDir + 'scan_error_detect_log.csv'):
        df_log = pd.read_csv(logDir + 'scan_error_detect_log.csv')
        df_update = pd.concat([df_log, df_log_], axis=0, ignore_index=True)
        df_update.to_csv(logDir + 'scan_error_detect_log.csv', index=False)
    else:
        if len(df_log_) > 0:
            df_log_.to_csv(logDir + 'scan_error_detect_log.csv', index=False)

    #Backup 'scan_error_detect_log' (10% of the time), garbage collection
    if randrange(10) == 7:
        df_log = pd.read_csv(logDir + 'scan_error_detect_log.csv')
        datetime_str = datetime.now().strftime("%Y%m%d_%Hh%M")
        os.makedirs(logDir + 'backups/', exist_ok=True)
        df_log.to_csv(logDir + 'backups/' + 'scan_error_detect_log-' + datetime_str + '.csv', index=False)
        collected = gc.collect()
        print("Garbage collector: collected",
                "%d objects." % collected)

    #Check stop conditions
    if len(subdir_ids_rem) < 2:
        print('Stop!')
        stop_condition = True
    if stop_condition_counter == stop_loop_threshold:
        print('Stop!')
        stop_condition = True


