'''
OCR post-processor for Alouette ionogram result CSVs.
Reads num2-type metadata from processed ionogram images using keras-ocr,
merges OCR results into existing result CSVs, and reclassifies image types accordingly.

Usage:
    python Alouette_processor2.py <rootDir> <user_prefix> <instance> <batch_size> <satellite_code>

    rootDir        : Root directory containing processed results and logs
    user_prefix    : Username prefix for logging (e.g. 'mgraff')
    instance       : Instance ID to distinguish parallel workers (e.g. '1')
    batch_size     : Number of images per OCR batch (e.g. 10)
    satellite_code : Two-digit satellite prefix to match in OCR string ('10' for Alouette-1, '20' for Alouette-2)
    gpu_env_path   : (Optional) Path to GPU TensorFlow environment site-packages to enable GPU processing
                     e.g. 'U:/temp/user/python/envs/tf210/lib/site-packages/'

Required folder structure:
    rootDir/
    ├── 03_processing/
    │   └── <directory>/<subdirectory>/   ← processed ionogram images
    ├── 05_result/
    │   └── <directory>/
    │       └── result-<directory>_<subdirectory>.csv
    └── 06_log/
        └── process_log_A.csv             ← master list of subdir_ids to process
'''

#OCR read 'num2' metadata

import sys
import warnings
warnings.filterwarnings('ignore')

# GPU environment path must be inserted before keras_ocr/tensorflow is imported
gpu_env_path = sys.argv[6] if len(sys.argv) > 6 else ''
if gpu_env_path:
    sys.path.insert(0, gpu_env_path)
    import tensorflow as tf
    print('TensorFlow version:', tf.__version__)
    print('GPU devices:', tf.config.list_physical_devices('GPU'))

import pandas as pd
import numpy as np
import os
from random import randrange
import time
from datetime import datetime
import gc

import keras_ocr
pipeline = keras_ocr.pipeline.Pipeline()
import cv2

#Set parameters
user_prefix = sys.argv[2]
instance = sys.argv[3]
user = user_prefix + instance #e.g: 'Rav Super2'
batch_size = int(sys.argv[4])
satellite_code = sys.argv[5]  # e.g. '10' for Alouette-1, '20' for Alouette-2
process_on_VDI = True
stop_loop_threshold = 6000 #max while loops to prevent infinite loop

#Set directories
rootDir = sys.argv[1]
processedDir = rootDir + '04_processed/'
resultDir = rootDir + '05_result/'
logDir = rootDir + '06_log/'


#Functions
def read_num2_metadata(prediction_groups, subdir_path, batch_i, img_fns, satellite_code):
    
    df_read = pd.DataFrame()
    df_notread = pd.DataFrame()
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
        
            #String concatenate, fix string
            read_str = ''
            for j in range(0, len(df_ocr)):
                read_str_ = df_ocr['number'].iloc[j]
                read_str += read_str_
            read_str = read_str.replace('o', '0')

            #Test for num2
            if len(read_str) == 15:
                if read_str[0:2] == satellite_code:
                    row2 = pd.DataFrame({
                        'station_number_OCR': read_str[2:4],
                        'year_OCR': read_str[4:6],
                        'day_of_year_OCR': read_str[6:9],
                        'hour_OCR': read_str[9:11],
                        'minute_OCR': read_str[11:13],
                        'second_OCR': read_str[13:15],
                        'filename': img_fns[batch_i + i].replace(subdir_path, '')
                    }, index=[i])
                    df_read = pd.concat([df_read, row2])
                else:
                    df_ocr['filename'] = img_fns[batch_i + i].replace(subdir_path, '')
                    df_notread = pd.concat([df_notread, df_ocr])
            else:
                df_ocr['filename'] = img_fns[batch_i + i].replace(subdir_path, '')
                df_notread = pd.concat([df_notread, df_ocr])
    
    return df_read, df_notread


def draw_random_subdir(processedDir, logDir):
    
    directory_list = os.listdir(processedDir)
    directory = directory_list[randrange(len(directory_list))]
    subdirectory_list = os.listdir(processedDir + directory + '/')
    subdirectory = subdirectory_list[randrange(len(subdirectory_list))]
    
    #Check randomly-selected directory and subdirectory against the 'process_log_OCR'
    if os.path.exists(logDir + 'process_log_OCR.csv'):
        df_log = pd.read_csv(logDir + 'process_log_OCR.csv')
        df_search = df_log.loc[(df_log['Directory'] == directory) & (df_log['Subdirectory'] == subdirectory)]
        if len(df_search) > 0:
            print(directory + '/' + subdirectory + ' already processed!')
            return '', ''
        else:
            return directory, subdirectory
    else:
        return directory, subdirectory



#Process remaining subdirectories with while loop
stop_condition = False
stop_condition_counter = 0

while not stop_condition:
    start = time.time()
    
    #Draw random, yet to be processed subdirectory, to process
    df_inventory = pd.read_csv(logDir + 'process_log_A.csv')
    subdir_ids_tot = df_inventory['subdir_id'].unique()
    if os.path.exists(logDir + 'process_log_OCR.csv'):
        df_log = pd.read_csv(logDir + 'process_log_OCR.csv')
        subdir_ids_proc = df_log['subdir_id'].unique()
    else:
        subdir_ids_proc = []
    subdir_ids_rem = list(set(subdir_ids_tot) - set(subdir_ids_proc))
    if len(subdir_ids_rem) > 0:
        sel = subdir_ids_rem[randrange(len(subdir_ids_rem))]
        directory, subdirectory = sel.split('_', 1)
    else:
        directory, subdirectory = draw_random_subdir(processedDir=processedDir, logDir=logDir)
    if not directory:
        if len(subdir_ids_rem) < 2:
            print('Stop!')
            stop_condition = True
        stop_condition_counter += 1
        if stop_condition_counter >= stop_loop_threshold:
            print('Stop!')
            stop_condition = True
        continue
    subdir_path_end = directory + '/' + subdirectory + '/'

    #Process subdirectory
    print('')
    print('Processing ' + subdir_path_end + ' subdirectory...')
    print(str(len(subdir_ids_rem)) + ' subdirectories to go!')
    img_fns = []
    for file in os.listdir(processedDir + subdir_path_end):
        path = processedDir + subdir_path_end + file
        if not os.path.isfile(path):
            print('Skipping broken image (not a file):', path)
            continue
        try:
            if os.path.getsize(path) == 0:
                print('Skipping broken image (zero size):', path)
                continue
        except Exception as e:
            print('Could not stat file, skipping:', path, 'error:', e)
            continue
        try:
            img = cv2.imread(path)
            if img is None:
                print('Skipping broken image (unreadable by OpenCV):', path)
                continue
        except Exception as e:
            print('Error reading image with OpenCV, skipping:', path, 'error:', e)
            continue
        img_fns.append(path)
    n_batches = int(np.floor(len(img_fns)/batch_size))
    batch_remainder = len(img_fns)%batch_size
    df_read = pd.DataFrame()
    df_notread = pd.DataFrame()
    for i in range(0, n_batches):
        print('Starting batch... ' + str(i))
        batch_i = i*batch_size
        batch_f = batch_i + batch_size
        try:
            prediction_groups = pipeline.recognize(img_fns[batch_i:batch_f])
            df_read_, df_notread_ = read_num2_metadata(prediction_groups=prediction_groups, subdir_path=processedDir + subdir_path_end, batch_i=batch_i, 
                                                       img_fns=img_fns, satellite_code=satellite_code)
            df_read = pd.concat([df_read, df_read_])
            df_notread = pd.concat([df_notread, df_notread_])
        except Exception as e:
            print('pipeline.recognize failed for batch', batch_i, batch_f, 'error:', e)
    #Remainder
    print('Finishing up...')
    if batch_remainder > 0:
        batch_i = n_batches*batch_size
        batch_f = batch_i + batch_remainder
        try:
            prediction_groups = pipeline.recognize(img_fns[batch_i:batch_f])
            df_read_, df_notread_ = read_num2_metadata(prediction_groups=prediction_groups, subdir_path=processedDir + subdir_path_end, batch_i=batch_i, 
                                                      img_fns=img_fns, satellite_code=satellite_code)
            df_read = pd.concat([df_read, df_read_])
            df_notread = pd.concat([df_notread, df_notread_])
        except Exception as e:
            print('pipeline.recognize failed for remainder', batch_i, batch_f, 'error:', e)
    
    #Integrate OCR read metadata into existing results data for subdirectory
    result_path = resultDir + directory + '/' + 'result-' + directory + '_' + subdirectory + '.csv'
    header_cols = [
        'Roll', 'Subdirectory', 'filename', 'processed_image_class', 'details',
        'station_number_OCR', 'year_OCR', 'day_of_year_OCR', 'hour_OCR', 'minute_OCR', 'second_OCR',
        'satellite_number', 'year', 'day_1', 'day_2', 'day_3', 'hour_1', 'hour_2', 'minute_1', 'minute_2',
        'second_1', 'second_2', 'station_number_1', 'station_number_2'
    ]
    if not os.path.exists(result_path):
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        pd.DataFrame(columns=header_cols).to_csv(result_path, index=False)
        print('Created empty result file with headers:', result_path)
    try:
        df_result = pd.read_csv(result_path)
        df_result = df_result.rename(columns={'Roll': 'Directory'})
    except Exception as e:
        print('Failed to read result file:', result_path, 'error:', e)
        df_result = pd.DataFrame()
    if len(df_result) > 0:
        if len(df_read) > 0:
            df_merge = df_result.merge(df_read, how='left', on='filename')
            mask = (
                (df_merge['processed_image_class'] != 'loss') &
                (df_merge['processed_image_class'] != 'outlier') &
                (df_merge['day_of_year_OCR'].notna())
            )
            df_merge.loc[mask, 'processed_image_class'] = 'num2'
        else:
            df_merge = df_result
    elif len(df_result) == 0 and len(df_read) > 0:
        print('WARNING: df_result is empty but df_read has data. Building df_merge from OCR results.')
        df_merge = df_read.copy()
        for col in header_cols:
            if col not in df_merge.columns:
                df_merge[col] = np.nan
        df_merge['processed_image_class'] = 'num2'
        df_merge['Directory'] = directory
        df_merge['Subdirectory'] = subdirectory
    else:
        df_merge = df_result
    
    #Classify rows with OCR letters read as 'loss' and clear metadata:
    OCR_cols = ['station_number_OCR', 'year_OCR', 'day_of_year_OCR', 'hour_OCR', 'minute_OCR', 'second_OCR']
    md_cols = ['satellite_number', 'year', 'day_1', 'day_2', 'day_3', 'hour_1', 'hour_2', 'minute_1', 'minute_2', 'second_1', 
           'second_2', 'station_number_1', 'station_number_2']
    if len(df_read) > 0:
        for col in OCR_cols:
            df_merge[col] = df_merge[col].astype('string')
            df_merge.loc[df_merge[col].str.contains("[a-zA-Z]"), 'processed_image_class'] = 'loss'
            df_merge.loc[df_merge[col].str.contains("[a-zA-Z]"), 'details'] = 'OCR read metadata contains letters'   
    n_OCR_read = 0
    if 'details' in df_merge.columns:
        loss_ocr_mask = (df_merge['processed_image_class'] == 'loss') & (df_merge['details'] == 'OCR read metadata contains letters')
        for col in md_cols:
            if col in df_merge.columns:
                df_merge.loc[loss_ocr_mask, col] = np.nan
    num2_mask = df_merge['processed_image_class'] == 'num2'
    for col in md_cols:
        if col in df_merge.columns:
            df_merge.loc[num2_mask, col] = np.nan
    n_OCR_read = int(num2_mask.sum())
    
    #If num2 metadata type is detected, classify images with all other metadata types as loss:
    if len(df_read) > 0:
        mask = df_merge['processed_image_class'].isin(['num', 'dot'])
        df_merge.loc[mask, 'processed_image_class'] = 'loss'
        if 'details' in df_merge.columns:
            df_merge.loc[mask, 'details'] = 'metadata could not be read by OCR'
        for col in md_cols:
            if col in df_merge.columns:
                df_merge.loc[mask, col] = np.nan
    
    #If num2 metadata type is not detected:
    if len(df_read) == 0:
        n_num = len(df_merge.loc[df_merge['processed_image_class'] == 'num'])
        n_dot = len(df_merge.loc[df_merge['processed_image_class'] == 'dot'])
        #If num type metadata is the majority, classify dot type images as loss:
        if n_num > n_dot:
            mask = df_merge['processed_image_class'] == 'dot'
            df_merge.loc[mask, 'processed_image_class'] = 'loss'
            if 'details' in df_merge.columns:
                df_merge.loc[mask, 'details'] = 'metadata was interpreted to be dot type'
            for col in md_cols:
                if col in df_merge.columns:
                    df_merge.loc[mask, col] = np.nan
        #If dot type metadata is the majority, classify num type images as loss:
        if n_dot > n_num:
            mask = df_merge['processed_image_class'] == 'num'
            df_merge.loc[mask, 'processed_image_class'] = 'loss'
            if 'details' in df_merge.columns:
                df_merge.loc[mask, 'details'] = 'metadata was interpreted to be num type'
            for col in md_cols:
                if col in df_merge.columns:
                    df_merge.loc[mask, col] = np.nan
    
    #Save:
    try:
        df_merge.to_csv(resultDir + directory + '/' + 'result_OCRpass-' + directory + '_' + subdirectory + '.csv', index=False)
    except Exception as e:
        print('Failed to save result CSV for', subdir_path_end, 'error:', e)
    
    end = time.time()
    t = end - start
    print('Processing time for subdirectory: ' + str(round(t/60, 1)) + ' min')
    if len(img_fns) > 0:
        print('Processing rate: ' + str(round(t/len(img_fns), 2)) + ' s/img')
    else:
        print('No valid images found in subdirectory.')
    print('')

    #Record performance
    df_result_ = pd.DataFrame({
        'Directory': directory,
        'Subdirectory': subdirectory,
        'Process_time': t,
        'Process_timestamp': datetime.fromtimestamp(end),
        'User': user,
        'subdir_id': directory + '_' + subdirectory
    }, index=[0])
    try:
        if os.path.exists(logDir + 'process_log_OCR.csv'):
            df_log = pd.read_csv(logDir + 'process_log_OCR.csv')
            df_update = pd.concat([df_log, df_result_], axis=0, ignore_index=True)
            df_update.to_csv(logDir + 'process_log_OCR.csv', index=False)
        else:
            if len(df_result_) > 0:
                df_result_.to_csv(logDir + 'process_log_OCR.csv', index=False)
    except Exception as e:
        print('Failed to update process_log_OCR.csv for', subdir_path_end, 'error:', e)

    #Backup 'process_log' (10% of the time), garbage collection
    if randrange(10) == 7:
        df_log = pd.read_csv(logDir + 'process_log_OCR.csv')
        datetime_str = datetime.now().strftime("%Y%m%d_%Hh%M")
        os.makedirs(logDir + 'backups/', exist_ok=True)
        df_log.to_csv(logDir + 'backups/' + 'process_log_OCR-' + datetime_str + '.csv', index=False)
        gc.collect()

    #Check stop conditions
    if len(subdir_ids_rem) < 2:
        print('Stop!')
        stop_condition = True
    if stop_condition_counter == stop_loop_threshold:
        print('Stop!')
        stop_condition = True