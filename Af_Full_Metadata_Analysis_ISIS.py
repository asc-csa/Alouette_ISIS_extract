
#ISIS Full MetaData Analysis - Jeyshinee Pyneeandee February 2024

#Required imports 
import pandas as pd
import os
from random import randrange
import time
from datetime import datetime
import gc
import warnings
warnings.filterwarnings('ignore')
import keras_ocr
from optparse import OptionParser

pipeline = keras_ocr.pipeline.Pipeline()
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
     my_path = logDir = 'ISIS_2_Directory_Subdirectory_List.csv'
elif options.isis == '3':
     #ISIS BATCH 3/RAW UPLOAD CHOSEN
     directory_path = 'L:/DATA/ISIS/raw_upload_20230421/'
     batch_size = 359
     my_path = logDir = 'ISIS_Raw_Upload_Directory_Subdirectory_List.csv'
else:
     #ISIS BATCH 1 
     directory_path = 'L:/DATA/ISIS/ISIS_101300030772/'
     batch_size = 1720
     my_path = logDir = 'ISIS_1_Directory_Subdirectory_List.csv'
     


stop_loop_threshold = 6000 #max while loops to prevent infinite loop

#Log Directory, do not change
logDir = 'L:/DATA/ISIS/ISIS_Test_Metadata_Analysis/04_log/'
#Path to save results, do not change
resultDir = 'L:/DATA/ISIS/ISIS_Test_Metadata_Analysis/05_results/'

######

def read_metadata(prediction_groups, subdir_path, img):
    '''
    Definition:
        This function reads the prediction groups produced by KERAS OCR and seperates the metadata strings 
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
    for i in range(0, len(prediction_groups)):
        df_ocr = pd.DataFrame()
        predicted_image = prediction_groups[i]
        if len(predicted_image) > 0:
            for text, box in predicted_image:
                row = pd.DataFrame({
                    'number': text,
                    'x': box[1][0],
                    'y': box[1][1]
                }, index=[0])
                df_ocr = pd.concat([df_ocr, row])
            df_ocr = df_ocr.sort_values('x').reset_index(drop=True)
        
            #String concatenate, get all metadata in one string
            read_str = ''
            for j in range(0, len(df_ocr)):
                read_str_ = df_ocr['number'].iloc[j]
                read_str += read_str_

            if len(read_str) == 15:
                    row2 = pd.DataFrame({'Satellite_code': read_str[0:1],
                                         'Fixed_Frequency_code': read_str[1:2],
                                         'Station_Number_1': read_str[2:3],
                                         'Station_Number_2': read_str[3:4],
                                         'Year': read_str[4:6],
                                         'Day_1': read_str[6:7],
                                         'Day_2': read_str[7:8],
                                         'Day_3': read_str[8:9],
                                         'Hour_1': read_str[9:10],
                                         'Hour_2': read_str[10:11],
                                         'Minute_1': read_str[11:12],
                                         'Minute_2': read_str[12:13],
                                         'Second_1': read_str[13:14],
                                         'Second_2': read_str[14:15],
                                         'Filename': img.replace(subdir_path, '')
                                         }, index=[i])
                    df_read_temp = pd.concat([df_read, row2])
            else:
                    df_ocr['Filename'] = img.replace(subdir_path, '')
                    df_notread_temp = pd.concat([df_notread_temp, df_ocr])
        else:
                df_ocr['Filename'] = img.replace(subdir_path, '')
                df_notread_temp = pd.concat([df_notread_temp, df_ocr])
    
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

                for ind in full_dir_df.index:
                    directory = full_dir_df['Directory'][ind]
                    subdir = full_dir_df['Subdirectory'][ind]

                    if (full_dir_df['Status'][ind]) == "Not Processed":
                        full_dir_df.loc[ind, "Status"] = "In Progress"
                        full_dir_df.to_csv(my_path, index=False)
                        return directory, subdir, ind

                    elif (full_dir_df['Status'][ind]) == "In Progress":
                        print('Current subdirectory', subdir, 'being processed already, moving on to the next one')
                        continue
                    
                    elif (full_dir_df['Status'][ind]) == "Processed":
                        print("Current subdirectory", subdir, "already processed, moving on to the next one")
                        continue

            except (OSError, PermissionError) as e:
                print(my_path, 'currently being used, pausing for 30 seconds before another attempt')
                time.sleep(30)
                
def update_my_log_file(ind):
    '''
    Definition:
      
    Arguments:
        ind:

    Returns:
        
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
                
#Process remaining subdirectories with while loop
stop_condition = False

while stop_condition == False:
    start = time.time()
    
    #Draw random, yet to be processed subdirectory, to process
    if os.path.exists(logDir + 'process_log_OCR.csv'):
        try:
            my_log_file = pd.read_csv(logDir + 'Process_Log.csv')
            subdirs_processed = len(my_log_file['Subdirectory'].drop_duplicates())
            dirs_processed = len(my_log_file['Directory'].drop_duplicates())

        except (OSError, PermissionError) as e:
            print(logDir + 'process_log_OCR.csv', 'currently being used, pausing for 30 seconds before another attempt')
            time.sleep(30)
                
        subdir_rem = batch_size - subdirs_processed
        #Check stop conditions
        if subdir_rem < 2:
            print('Stop!')
            stop_condition = True

        # if stop_condition_counter == stop_loop_threshold:
        #     print('Stop!')
        #     stop_condition = True


    #Process subdirectory
    print('')
    print('Processing ' + subdir_path_end + ' subdirectory...')
    print(str(subdir_rem) + ' subdirectories to go!')


    #Get directory and subdirectory path to process and current row index
    directory, subdirectory, curr_row_index = draw_random_subdir()
    subdir_path_end = directory + '/' + subdirectory + '/'

    #Get all images from chosen directory and subdirectory path
    img_fns = []
    for file in os.listdir(directory_path + subdir_path_end):
        img_fns.append(directory_path + subdir_path_end + file)

    df_read = pd.DataFrame()
    df_notread = pd.DataFrame()

    for img in img_fns:
        prediction_groups = pipeline.recognize(img) ### TO EDIT BASED ON KERAS CODE 
        #############################################################################
        #############################################################################
        #############################################################################

        df_read_, df_notread_ = read_metadata(prediction_groups=prediction_groups, subdir_path=directory_path + subdir_path_end,
                                                       img=img)
        df_read = pd.concat([df_read, df_read_])
        df_notread = pd.concat([df_notread, df_notread_])
        
    #Saving results:
    my_temp_path = resultDir + directory
    if not os.path.notexists(my_temp_path):
        os.makedirs(resultDir,directory)
    
    df_read.to_csv(resultDir + directory + '/' +  'Metadata_analysis_' + subdirectory + '.csv', index=False)
    if len(df_notread) > 0:
        df_notread.to_csv(resultDir + directory + '/' +  'LOSS_Metadata_analysis_' + subdirectory + '.csv', index=False)

    #update status for current path of dir and subdir
    update_my_log_file(curr_row_index)
    print(directory, subdirectory, "processed - Status updated!")

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
            df_result_.to_csv(logDir + 'process_log_OCR.csv', index=False)
            
    #Backup 'process_log' (10% of the time), garbage collection
    if randrange(10) == 7:
        df_log = pd.read_csv(logDir + 'process_log_OCR.csv')
        datetime_str = datetime.now().strftime("%Y%m%d_%Hh%M")
        os.makedirs(logDir + 'backups/', exist_ok=True)
        df_log.to_csv(logDir + 'backups/' + 'process_log_OCR-' + datetime_str + '.csv', index=False)
        gc.collect()

    # #Check stop conditions
    # if len(subdir_ids_rem) < 2:
    #     print('Stop!')
    #     stop_condition = True
    # if stop_condition_counter == stop_loop_threshold:
    #     print('Stop!')
    #     stop_condition = True