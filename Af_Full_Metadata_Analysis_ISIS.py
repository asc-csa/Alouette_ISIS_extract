#ISIS Full MetaData Analysis - Jeyshinee Pyneeandee February 2024

#Required imports 
import pandas as pd
import os, gc, sys
from random import randrange
import time, datetime
from optparse import OptionParser

#from Ag_Keras_Ocr import *

#GPU not being used - need to fix 
sys.path.insert(0,  "c:/Users/jpyneeandee/.conda/envs/my_env_local/Lib/site-packages")

import tensorflow as tf
import keras_ocr

print('tensorflow version (should be 2.10.* for GPU compatibility)', tf.__version__)
if len(tf.config.list_physical_devices('GPU')) != 0:
    print('GPU in use for tensorflow')
else:
    print('CPU in use for tensorflow')


#To run this script : e.g Af_Full_Metadata_Analysis_ISIS.py --username jpyneeandee --isis 1 (for ISIS batch 1 run by Jeyshinee)
#Script defaults to ISIS Batch 1 

parser = OptionParser()

parser.add_option('-u', '--username', dest='username', 
        default='jpyneeandee', type='str', 
        help='CSA Network username, default=%default.')

parser.add_option('--isis', dest='isis', 
        default='1', type='str', 
        help='ISIS batch, default=%default.')

(options, args) = parser.parse_args()

        
if options.isis == '2':
     #ISIS BATCH 2 CHOSEN
     directory_path = 'L:/DATA/ISIS/ISIS_102000056114/'
     batch_size = 801

    #Log Directory, do not change
     logDir = 'L:/DATA/ISIS/ISIS_Test_Metadata_Analysis/BATCH_2/04_log/'
     #Path to save results, do not change
     resultDir = 'L:/DATA/ISIS/ISIS_Test_Metadata_Analysis/BATCH_2/05_results/'
     my_path = logDir + 'ISIS_2_Directory_Subdirectory_List.csv'

elif options.isis == '3':
     #ISIS BATCH 3/RAW UPLOAD CHOSEN
     directory_path = 'L:/DATA/ISIS/raw_upload_20230421/'
     batch_size = 359

    #Log Directory, do not change
     logDir = 'L:/DATA/ISIS/ISIS_Test_Metadata_Analysis/BATCH_3/04_log/'
     #Path to save results, do not change
     resultDir = 'L:/DATA/ISIS/ISIS_Test_Metadata_Analysis/BATCH_3/05_results/'
     my_path = logDir + 'ISIS_Raw_Upload_Directory_Subdirectory_List.csv'

else:
     #ISIS BATCH 1 
     directory_path = 'L:/DATA/ISIS/ISIS_101300030772/'
     batch_size = 1720

    #Log Directory, do not change
     logDir = 'L:/DATA/ISIS/ISIS_Test_Metadata_Analysis/BATCH_1/04_log/'
     #Path to save results, do not change
     resultDir = 'L:/DATA/ISIS/ISIS_Test_Metadata_Analysis/BATCH_1/05_results/'
     my_path = logDir + 'ISIS_1_Directory_Subdirectory_List.csv'

#station names and location 
station_log_dir = 'L:/DATA/ISIS/ISIS_Test_Metadata_Analysis/Station_Number_Name_Location.csv'
station_df = pd.read_csv(station_log_dir)

#KERAS OCR script to crop the metadata part of the ionogram, remove noise and white line
#This script also applies KERAS OCR to read the metadata and uses a recognizer trained on denoised ISIS ionograms. 

#Required imports 
import os
import string 
from PIL import Image
import keras_ocr
import string
import tempfile

#Loading trained recognizer model
recognizer = keras_ocr.recognition.Recognizer(alphabet= string.digits) 
recognizer.model.load_weights('L:/DATA/ISIS/keras_ocr_training/ISIS_reading_final.h5')  
recognizer.compile()  
pipeline = keras_ocr.pipeline.Pipeline(recognizer=recognizer)

#Paramaters for denoising code
imageHeight = 50
top_noise_height = 10
bottom_noise_height = 10
threshold_toLine=(110, 110, 110, 255)
threshold_towhite=(0, 0, 0, 255)
threshold_toblack=(80, 80, 80, 255)
start_row_to_process = 1
end_row_to_process = 20

#Functions
def crop_and_copy(input_path, output_path, imageHeight):
    # Open the input image
    with Image.open(input_path) as img:
        # Get the dimensions of the image
        width, height = img.size
        # Define the region to crop (imageHeight pixels from the bottom)
        crop_region = (0, height - imageHeight, width, height)
        # Crop the image
        cropped_img = img.crop(crop_region)
        # Create a new image with the same size as the cropped region
        new_img = Image.new("RGBA", (width, imageHeight), (0, 0, 0, 0))
        # Paste the cropped region onto the new image
        new_img.paste(cropped_img, (0, 0))
        # Save the result to the output path
        new_img.save(output_path.name)

def remove_top_bottom_noise(input_path, top_noise_height, bottom_noise_height):
    # Open the image
    with Image.open(input_path) as img:
        # Get the dimensions of the image
        width, height = img.size
        # Create a new image with the same content as the original
        new_img = img.copy()

        # Add a black border to the top noise height
        for y in range(top_noise_height):
            for x in range(width):
                new_img.putpixel((x, y), (0, 0, 0, 255))  # Set pixel to black
        # Remove noise from the bottom
        for y in range(height - bottom_noise_height, height):
            for x in range(width):
                new_img.putpixel((x, y), (0, 0, 0, 255))  # Set pixel to black

        # Save the result, overwriting the original image
        new_img.save(input_path.name)

def process_middle_lines_noise(input_path, threshold_toline, start_row, end_row):
    # Open the image
    img = Image.open(input_path)
    
    # Get the pixels
    pixels = img.load()
    width, _ = img.size
    # Iterate through rows to process and replace colors below the threshold
    for y in range(start_row, end_row + 1):
        for x in range(width):
            r, g, b, a = pixels[x, y]
            if (r, g, b, a) < threshold_toline:
                pixels[x, y] = (0, 0, 0, 255)
    # Iterate through all rows to process the below threshold rest pixels to black
    for y in range(top_noise_height, imageHeight-bottom_noise_height):
        if y == 19 or y == 20:
           continue
        for x in range(width):
            r, g, b, a = pixels[x, y]
            if (r, g, b, a) < threshold_toblack:
                pixels[x, y] = (0, 0, 0, 255)
    # Iterate through all rows to process the rest pixels to white
    for y in range(top_noise_height, imageHeight-bottom_noise_height):
        for x in range(width):
            r, g, b, a = pixels[x, y]
            if (r, g, b, a) > threshold_towhite:
                pixels[x, y] = (255, 255, 255, 255)
    # Save the modified image
    img.save(input_path.name)

def read_image(image_path):
    try: 
        #applying cropping and de-noising filters
        output_file_path = tempfile.NamedTemporaryFile(delete = False, suffix=".png")
        crop_and_copy(image_path, output_file_path, imageHeight)
        remove_top_bottom_noise(output_file_path,top_noise_height,bottom_noise_height)

        process_middle_lines_noise(output_file_path,threshold_toLine,start_row_to_process,end_row_to_process)

        #reading filtered image
        image = keras_ocr.tools.read(output_file_path.name) 
      
        #applying keras
        prediction = pipeline.recognize([image])[0]
    
        combined_lists = list(zip([x[1][0][0] for x in prediction], [x[0] for x in prediction]))
        sorted_lists = sorted(combined_lists, key=lambda x: x[0])
        sorted_digits = [item[1] for item in sorted_lists]
        print("Metadata read:", sorted_digits)

    except Exception as e:
        print('ERR:', e)

    output_file_path.close()
    os.unlink(output_file_path.name)

    return sorted_digits

     
#Functions

def read_metadata(prediction_groups, subdir_path, img):
    '''
    Definition:
        This function reads the prediction groups produced by KERAS OCR and separates the metadata strings 
        into their appropriate tags
    Arguments:
        prediction_groups: KERAS OCR output predictions
        subdir_path: Directory + subdirectory path to image
        img: Image being processed 
    Returns:
        Two dataframes of processed metadata and loss images (metadata that have less than 15 characters)
    '''
    #get metadata from keras prediction & concat string
    df_read_temp = pd.DataFrame()
    df_notread_temp = pd.DataFrame()

    read_str = str(prediction_groups)
    if len(read_str) == 15:
            station_number = read_str[2:4]
            station_location, station_lat, station_lon, station_ID = get_station_info(station_number)
            row2 = pd.DataFrame({'Satellite_Code': read_str[0:1],
                                'Fixed_Frequency_Code': read_str[1:2],
                                'Station_Number': station_number,
                                'Station_Location':station_location,
                                'Station_ID':station_ID,
                                'Station_Lat':station_lat,
                                'Station_Lon':station_lon,
                                'Year': read_str[4:6],
                                'Day': read_str[6:9],
                                'Hour': read_str[9:11],
                                'Minute': read_str[11:13],
                                'Second': read_str[13:15],
                                'Filename': img.replace(subdir_path, '')
                                         })
            df_read_temp = pd.concat([df_read_temp, row2])
    else:
            df_notread_temp.loc[0,'Filename'] = img.replace(subdir_path, '')
    
    return df_read_temp, df_notread_temp

def draw_random_subdir():
    '''
    Definition: Draw a directory and subdirectory, that is not currently in progress or has already been processed and updates the status
    of that row 
      
    Arguments: None

    Returns: A directory, subdirectory and the row number at which these are found 
        
    '''
    if os.path.exists(my_path):
            try:
                full_dir_df = pd.read_csv(my_path)
                ind = randrange(len(full_dir_df))
            #   for ind in full_dir_df.index: #instead of a for loop, i'm trying a randomized ind
                directory = full_dir_df['Directory'][ind]
                subdir = full_dir_df['Subdirectory'][ind]

                if (full_dir_df['Status'][ind]) == "Not Processed":
                    full_dir_df.loc[ind, "Status"] = "In Progress"
                    full_dir_df.to_csv(my_path, index=False)
                    return directory, subdir, ind

                elif (full_dir_df['Status'][ind]) == "In Progress":
                    print('Current subdirectory', subdir, 'being processed already, moving on to the next one')
                    draw_random_subdir()
                    
                elif (full_dir_df['Status'][ind]) == "Processed":
                    print("Current subdirectory", subdir, "already processed, moving on to the next one")
                    draw_random_subdir()

            except (OSError, PermissionError) as e:
                print(my_path, 'currently being used, pausing for 30 seconds before another attempt')
                time.sleep(30)

                
def update_my_log_file(ind):
    '''
    Definition: Updates the status of the given index row for dir and subdir as "Processed"
      
    Arguments:
        ind: Index row for processed directory and subdirectory

    Returns: None
        
    '''
    if os.path.exists(my_path):
        try:
            full_dir_df = pd.read_csv(my_path)
            
            if (full_dir_df['Status'][ind]) == "In Progress":
                    full_dir_df.loc[ind, "Status"]= "Processed"
                    full_dir_df.to_csv(my_path, index=False)
            else:
                print("Error with this path - check if dir and subdir at row", str(ind), "has been processed")
                 
        except (OSError, PermissionError) as e:
                print(my_path, 'currently being used, pausing for 30 seconds before another attempt')
                time.sleep(30)


def get_station_info(ind):
    '''
    Definition: This function takes in a station number (read from an ionogram), cross references it with 
    a station information csv and returns the station location, ID, Latitude and Longitude 
 
    Arguments:
        ind: int corresponding to station number 

    Returns: 4 strings corresponding to the location, latitude, longitiude and ID of the given station number 

    '''    
    for i in range(len(station_df)):
        if station_df['Number'][i] == str(ind):
            station_location = station_df['Location'][i]
            station_lat =  station_df['Latitude'][i]
            station_lon = station_df['Longitude'][i]
            station_ID =  station_df['Station ID'][i]
        else:
            station_ID = station_location = station_lon = station_lat = 0

    return station_location, station_lat, station_lon, station_ID

#Process remaining subdirectories with while loop
stop_condition = False

while stop_condition == False:
    start = time.time()
    
    #Get number of processed subdirs 
    if os.path.exists(logDir + 'Process_Log.csv'):
        try:
            my_log_file = pd.read_csv(logDir + 'Process_Log.csv')
            subdirs_processed = len(my_log_file['Subdirectory'].drop_duplicates())
            dirs_processed = len(my_log_file['Directory'].drop_duplicates())

        except (OSError, PermissionError) as e:
            print(logDir + 'Process_Log_OCR.csv', 'currently being used, pausing for 30 seconds before another attempt')
            time.sleep(30)

        #get remaining subdirs        
        subdir_rem = batch_size - subdirs_processed

        #Check stop conditions
        if subdir_rem < 2:
            print('Stop!')
            stop_condition = True

   
    #Get directory and subdirectory path to process and current row index
    directory, subdirectory, curr_row_index = draw_random_subdir()
    subdir_path_end = directory + '/' + subdirectory + '/'

    #Process subdirectory
    print('')
    print('Processing ' + subdir_path_end + ' subdirectory')
    print(str(subdir_rem) + ' subdirectories to go!')

    #Get all images from chosen directory and subdirectory path
    img_fns = []
    for file in os.listdir(directory_path + subdir_path_end):
        img_fns.append(directory_path + subdir_path_end + file)
        num_images = len(img_fns)

    df_read = pd.DataFrame()
    df_notread = pd.DataFrame()

    for img in img_fns:
        prediction_groups = read_image(img) 
        df_read_, df_notread_ = read_metadata(prediction_groups=prediction_groups, subdir_path=directory_path + subdir_path_end,
                                                       img=img)
        df_read = pd.concat([df_read, df_read_])
        df_notread = pd.concat([df_notread, df_notread_])
        
    #Saving results:
    my_temp_path = resultDir + directory
    if not os.path.exists(my_temp_path):
        path = os.path.join(resultDir, directory)
        os.makedirs(path)
    
    df_read.to_csv(resultDir + directory + '/' +  'Metadata_analysis_' + subdirectory + '.csv', index=False)
    if len(df_notread) > 0:
        df_notread.to_csv(resultDir + directory + '/' +  'LOSS_Metadata_analysis_' + subdirectory + '.csv', index=False)

    print('Dir:', directory, 'Subdir:', subdirectory, "results saved to csv!")

    #update status for current path of dir and subdir
    update_my_log_file(curr_row_index)
    print("Status updated!")

    #Processing time for one subdirectory
    end = time.time()
    t = end - start
    print('Processing time for subdirectory: ' + str(round(t/60, 1)) + ' min')
    print('Processing rate: ' + str(round(t/len(img_fns), 2)) + ' s/img')
    print('')

    #Record performance
    df_result_ = pd.DataFrame({
        'Directory': directory,
        'Subdirectory': subdirectory,
        '# images' : num_images,
        'Process_time': t,
        'Process_timestamp': datetime.fromtimestamp(end),
        'User': options.username
    }, index=[0])

    if os.path.exists(logDir + 'Process_Log.csv'):
        df_log = pd.read_csv(logDir + 'Process_Log.csv')
        df_update = pd.concat([df_log, df_result_], axis=0, ignore_index=True)
        df_update.to_csv(logDir + 'Process_Log.csv', index=False)

    else:
        if len(df_result_) > 0:
            df_result_.to_csv(logDir + 'Process_Log_OCR.csv', index=False)
            
    #Backup 'process_log' (10% of the time), garbage collection
    if randrange(10) == 7:
        df_log = pd.read_csv(logDir + 'Process_Log_OCR.csv')
        datetime_str = datetime.now().strftime("%Y%m%d_%Hh%M")
        os.makedirs(logDir + 'backups/', exist_ok=True)
        df_log.to_csv(logDir + 'backups/' + 'process_log_OCR-' + datetime_str + '.csv', index=False)
        gc.collect()
