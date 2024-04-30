
#Jeyshinee Pyneeandee - Nov 2023 - Flagging Over Exposed ISIS Ionograms

#imports 
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from skimage.color import rgb2gray
from skimage.exposure import histogram
from skimage.util import img_as_ubyte
import os
import random
import pandas as pd
import time 
import gc

#Path to save results 
outFile = 'L:/DATA/ISIS/OverExposure/Batch2.csv'
#ISIS Ionograms Directory
batchDir = 'L:/DATA/ISIS/ISIS_102000056114/' 
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
        # Get intensity values for grayscale images or convert to grayscale
        img = imageio.imread(image_path)
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = rgb2gray(img)

        image_intensity = img_as_ubyte(img)
        freq, bins = histogram(image_intensity)
        width, height = img.shape[0], img.shape[1]
        total_pixels = width * height

        # Plot histogram, if true
        if plotting_hist:
            plt.step(bins, freq * 1.0 / freq.sum())
            plt.xlabel('intensity value')
            plt.ylabel('Fraction of pixels')
            plt.show()

        # Get histogram integral values for pixels >= 230
        integral_255 = 0
        for i in range(230, len(bins) - 1):
            bin_width = bins[i + 1] - bins[i]

            # Sum over the number in each bin and multiply by bin width
            integral_255 = integral_255 + (bin_width * sum(freq[i:i + 1]))

        proportion = integral_255 / total_pixels

        # Flag ionogram if prop 0.11
        if proportion > 0.11:
            return proportion, True
        else:
            return proportion, False

    except (ValueError, RuntimeError) as e:
        error_message = f"Error processing image '{image_path}': {e}"
        print(error_message)
        # Continue processing other images even if one fails
        return 0, False


def read_all_directories(outFile=outFile, append2outFile=True, batchDir=batchDir, plotting=False):

    header = True
    # set column types for results dataframe 
    types = {'Directory': 'str', 'Subdirectory': 'str', 'filename': 'str'}

    # initialize lists to save values to in loop
    directories, subdirs, images = [], [], []
    proportion_list = []

    raw_contents = os.listdir(batchDir) # random shuffle applied
    for directory in raw_contents:

            # loop over all subdirectories within the directory
            directory_contents = os.listdir(batchDir + directory) 
            for subdir in directory_contents:
                
                print('###############################')
                print('Directory:', directory, '\nSubdirectory:', subdir)
                print('###############################')

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
                            read_safe = True 

                        except (OSError, PermissionError) as e:
                            print(e)
                            print(outFile, 'currently being used, pausing for 30 seconds before another attempt')
                            time.sleep(30)

                    #  check that this specific subdir has already been processed
                    if (df_processed_results["Directory"].eq(directory).any()) and (df_processed_results["Subdirectory"].eq(subdir).any()):
                        print(str(directory + ' ' + subdir))
                        print('This subdirectory has already been processed, moving on to the next one')
                        proceed = False # the continue statement was not working as expected
                
                    else:
                        print('This subdirectory has not already been processed, beginning processing')
                        proceed = True
                
                else: # TEMP
                    print(f'error: OutFile does not exist ({outFile}), did not save!!')
                    proceed = False

                del df_processed_results
                
                if proceed == True:
                    #add directory and subdirectory names to file, for future
                    # check in case no ionogram was flagged
                    subdir_temp = []
                    subdir_temp.append(subdir)
                    dir_temp = []
                    dir_temp.append(directory)
                    temp_subdir = pd.DataFrame()
                    temp_subdir['Directory'] = dir_temp
                    temp_subdir['Subdirectory'] = subdir_temp
                    temp_subdir.to_csv(outFile, mode='a', index=False, header=header)
                    del subdir_temp
                    del temp_subdir
                    del dir_temp
                    header = True

                    # loop over all images in the subdirectory
                    subdir_contents = os.listdir(batchDir + directory + '/' + subdir) 
                    for image in subdir_contents:
                        # save full path of image
                         if image.endswith(".png"):
                            image_path = batchDir + directory + '/' + subdir + '/' + image       

                            # Pass image to flag_overexposed to check for overexposure
                            prop_val, bool_check = flag_overexposed(image_path, plotting_hist = plotting)

                            if bool_check:
                                directories.append(directory)
                                subdirs.append(subdir)
                                images.append(image)
                                proportion_list.append(prop_val)


                    ### SAVE RESULTS AFTER PROCESSING EACH SUBDIR ####                
                    # initialize dataframe and save results to csv
                    # (redoing this each interation to not loose information)
                    df_mapping_results = pd.DataFrame()
                    df_mapping_results['Directory'] = directories
                    df_mapping_results['Subdirectory'] = subdirs
                    df_mapping_results['filename'] = images
                    df_mapping_results['Proportion of Overexposure'] = proportion_list


                    # mode = 'a' means it will append to existing data within the file
                    if append2outFile == True:
                        mode = 'a' 

                        # wipe lists now that they have been saved
                        directories, subdirs, images = [], [], []
                        proportion_list = []
                        
                    else: # append2outFile = False should not be used for multi-instance
                        # this overwrites existing file
                        mode = 'w' # check exits or not instead?
                        header = True

                    print('Subdirectory', subdir, 'processed, now attempting to save results')
                    save = True
                    print(os.path.exists(outFile))
                    if os.path.exists(outFile):
                        write_safe = False
                        while write_safe == False:
                            print('***')
                            try:
                                os.rename(outFile, outFile)
                                
                                if save == True:
                                    # check if there is already data in the output file 
                                    # (this may create duplicate headers if instances finish 
                                    # processing their first subdir too close together)
                                    if header == True and os.path.exists(outFile) and os.path.getsize(outFile)!=0:
                                        header = False 

                                    df_mapping_results.to_csv(outFile, mode=mode, index=False, header=header)
                                    del df_mapping_results
                                    print(f'Data sucessfully saved to {outFile}')

                                write_safe = True 

                            except (OSError, PermissionError) as e:
                                print(e)
                                print(outFile, 'Currently being used, pausing for 1 minute before another attempt')
                                time.sleep(60)

                    gc.collect()
                    #print("Garbage collector: collected", "%d objects." % collected)


if __name__ == '__main__':
    read_all_directories(plotting=False)

