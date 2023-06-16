# run these pip commands in anaconda prompt to download non-standard libraries
# >pip install tensorflow --user
# >pip install keras_ocr --user

# enter your network username to have correct paths
username = 'aferreira'
tf210_env  = '/python/envs/tf210/lib/site-packages/'

# imports
import sys
import cv2
import os
import gc
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from threading import Thread
from optparse import OptionParser
parser = OptionParser()

# replace this with your own library path for --user pip installs
sys.path.append('C:/Users/' + username + '/AppData/Roaming/Python/Python38/Scripts')

parser.add_option('-g', '--gpu_use', dest='gpu_use', 
        default=False, type='bool', 
        help='True to use GPU, False for CPU, default=%default.')

(options, args) = parser.parse_args()
if options.gpu_use:
    sys.path.insert(0, 'u:/temp/' + username + tf210_env)

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
save_dir = 'U:/Downloads/test_runs/' 
outFile = save_dir + 'notebook20_outputs_v17.csv'

# make the directory to save into  
# if it doesn't already exist
if not(os.path.exists(save_dir)):
    os.makedirs(save_dir)

# set default saving settings
# (not sure if code works anymore if these are changed)
append2outFile = True 
saveImages = False


def read_image(image_path, plotting=False, just_digits=False, down_factor=1):
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

        down_factor (int, optional): factor by which to integer divide height
                                     and width to scale down size of image

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
        #cropped_image = [image[cropped_height:height,:]]
        #^no longer cropping since we are looking for ISIS text

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
                #if just_digits == False or (just_digits == True and value.isdigit()):
                if True: # do not require it to be an integer, too much of a cut it seems

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


def read_all_rolls(outFile=outFile, append2outFile=True, batchDir=batchDir, plotting=False, max_images=None, save_each=100):
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

      max_images (int, optional): maximum number of images used to iterate over

      save_each (int, optional): save results to CSV after this number of images

   Returns:

      None

   '''
   # check if there is already data in the output file 
   if os.path.exists(outFile) and os.path.getsize(outFile)!=0:
      found = False
      header = False 


      df = pd.read_csv(outFile)
      last_entry = batchDir + df['roll'].iloc[-1] + '/' + df['subdir'].iloc[-1] + '/' + df['image'].iloc[-1]
      del df 

      # garbage collector
      collected = gc.collect()
      print("Garbage collector: collected",
               "%d objects." % collected)

   else: 
      found = True
      header = True
      last_entry = ''

   # initialize lists to save values to in loop
   rolls, subdirs, images = [], [], []
   heights, widths, digit_counts = [], [], []
   says_isis_lst = []

   images_saved = 0
   
   # loop over all rolls in the batch 2 raw data directory
   raw_contents = os.listdir(batchDir)
   for roll in raw_contents:

      # loop over all subdirectories within the roll
      roll_contents = os.listdir(batchDir + roll) 
      for subdir in roll_contents:
         
         # loop over all images in the subdirectory
         subdir_contents = os.listdir(batchDir + roll + '/' + subdir) 
         for image in subdir_contents:

            # save full path of image
            image_path = batchDir + roll + '/' + subdir + '/' + image

            # skip over image if already analyzed in CSV
            if found == False and last_entry == image_path:
               found = True

            if found == True:
               images_saved += 1

               if max_images != None and images_saved > max_images:
                  sys.exit()

               # save id of image
               rolls.append(roll)
               subdirs.append(subdir)
               images.append(image)

               # send to read_image to get aspect ratio, digit count, and isis text
               num_of_digits, h, w, says_isis = read_image(image_path)

               # save values
               digit_counts.append(num_of_digits)
               heights.append(h)
               widths.append(w)
               says_isis_lst.append(says_isis)


               # save to csv after a set number of images (perhaps best to make propto max images)
               if images_saved % save_each == 0:

                  # initialize dataframe and save results to csv
                  # (redoing this each interation to not loose information)
                  df_mapping_results = pd.DataFrame()

                  df_mapping_results['roll'] = rolls
                  df_mapping_results['subdir'] = subdirs
                  df_mapping_results['image'] = images
                  df_mapping_results['digit_count'] = digit_counts
                  df_mapping_results['height'] = heights
                  df_mapping_results['width'] = widths
                  df_mapping_results['says_isis'] = says_isis_lst

                  # mode = 'a' means it will append to existing data within the file
                  if append2outFile == True:
                     mode = 'a' 

                     # wipe lists now that they have been saved
                     rolls, subdirs, images = [], [], []
                     heights, widths, digit_counts = [], [], []
                     says_isis_lst = []
                     
                  else: 
                     # this overwrites existing file
                     mode = 'w'
                     header = True

                  df_mapping_results.to_csv(outFile, mode=mode, index=False, header=header)
                  del df_mapping_results

                  collected = gc.collect()
                  print("Garbage collector: collected",
                           "%d objects." % collected)


if __name__ == '__main__':
    read_all_rolls()