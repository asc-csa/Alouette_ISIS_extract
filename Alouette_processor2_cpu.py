#OCR read 'num2' metadata

import sys
sys.path.insert(0, "u:/temp/rnaidoo/python/envs/alouette_on_ravsupervdi2/lib/site-packages/")

import tensorflow as tf
print(tf.__version__)

with tf.device('/CPU:0'):

    import pandas as pd
    import numpy as np
    import os
    from random import randrange
    import time
    from datetime import datetime

    import warnings
    warnings.filterwarnings('ignore')

    import keras_ocr
    pipeline = keras_ocr.pipeline.Pipeline()

    #Set parameters
    instance = sys.argv[1]
    user = 'Rav Super' + instance #e.g: 'Rav Super2'
    batch_size = int(sys.argv[2])
    process_on_VDI = True
    stop_loop_threshold = 3000 #max while loops to prevent infinite loop

    #Set directories
    rootDir = 'L:/DATA/Alouette_I/BATCH_II_Run2/'
    processedDir = rootDir + '04_processed/'
    resultDir = rootDir + '05_result/'
    logDir = rootDir + '06_log/'


    #Functions
    def read_num2_metadata(prediction_groups, subdir_path, batch_i, img_fns):
        
        df_read = pd.DataFrame()
        df_notread = pd.DataFrame()
        for i in range(len(prediction_groups)):
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
                for j in range(len(df_ocr)):
                    read_str_ = df_ocr['number'].iloc[j]
                    read_str += read_str_
                read_str = read_str.replace('o', '0')

                #Test for num2
                if len(read_str) == 15 and read_str[:2] == '10':
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
        return df_read, df_notread


    def draw_random_subdir(processedDir, logDir):
        
        directory_list = os.listdir(processedDir)
        directory = directory_list[randrange(len(directory_list))]
        subdirectory_list = os.listdir(processedDir + directory + '/')
        subdirectory = subdirectory_list[randrange(len(subdirectory_list))]

        if not os.path.exists(f'{logDir}process_log_OCR.csv'):
            return directory, subdirectory
        df_log = pd.read_csv(f'{logDir}process_log_OCR.csv')
        df_search = df_log.loc[(df_log['Directory'] == directory) & (df_log['Subdirectory'] == subdirectory)]
        if len(df_search) <= 0:
            return directory, subdirectory
        print(f'{directory}/{subdirectory} already processed!')
        return '', ''



    #Process remaining subdirectories with while loop
    stop_condition = False
    stop_condition_counter = 0

    while stop_condition == False:
        start = time.time()
        
        #Draw random, yet to be processed subdirectory, to process
        df_inventory = pd.read_csv(logDir + 'image_inventory_processed.csv')
        subdir_ids_tot = df_inventory['subdir_id'].unique()
        if os.path.exists(logDir + 'process_log_OCR.csv'):
            df_log = pd.read_csv(logDir + 'process_log_OCR.csv')
            subdir_ids_proc = df_log['subdir_id'].unique()
        else:
            subdir_ids_proc = []
        subdir_ids_rem = list(set(subdir_ids_tot) - set(subdir_ids_proc))
        directory, subdirectory = draw_random_subdir(processedDir=processedDir, logDir=logDir)
        if len(directory) == 0:
            continue
        subdir_path_end = directory + '/' + subdirectory + '/'

        #Process subdirectory
        print('')
        print('Processing ' + subdir_path_end + ' subdirectory...')
        print(str(len(subdir_ids_rem)) + ' subdirectories to go!')
        img_fns = []
        for file in os.listdir(processedDir + subdir_path_end):
            img_fns.append(processedDir + subdir_path_end + file)
        n_batches = int(np.floor(len(img_fns)/batch_size))
        batch_remainder = len(img_fns)%batch_size
        df_read = pd.DataFrame()
        df_notread = pd.DataFrame()
        for i in range(n_batches):
            print('Starting batch... ' + str(i))
            batch_i = i*batch_size
            batch_f = batch_i + batch_size
            prediction_groups = pipeline.recognize(img_fns[batch_i:batch_f])
            df_read_, df_notread_ = read_num2_metadata(prediction_groups=prediction_groups, subdir_path=processedDir + subdir_path_end, batch_i=batch_i, 
                                                       img_fns=img_fns)
            df_read = pd.concat([df_read, df_read_])
            df_notread = pd.concat([df_notread, df_notread_])
        #Remainder
        print('Finishing up...')
        if batch_remainder > 0:
            batch_i = n_batches*batch_size
            batch_f = batch_i + batch_remainder
            prediction_groups = pipeline.recognize(img_fns[batch_i:batch_f])
            df_read_, df_notread_ = read_num2_metadata(prediction_groups=prediction_groups, subdir_path=processedDir + subdir_path_end, batch_i=batch_i, 
                                                      img_fns=img_fns)
            df_read = pd.concat([df_read, df_read_])
            df_notread = pd.concat([df_notread, df_notread_])
        
        #Integrate OCR read metadata into existing results data for subdirectory
        df_result = pd.read_csv(resultDir + directory + '/' + 'result-' + directory + '_' + subdirectory + '.csv')
        #Change 'Roll' to 'Directory':
        df_result = df_result.rename(columns={
            'Roll': 'Directory'
        })
        if len(df_result) > 0:
            if len(df_read) > 0:
                df_merge = df_result.merge(df_read, how='left', on='filename')
                for i in range(len(df_merge)):
                    if df_merge['processed_image_class'].iloc[i] != 'loss' and df_merge['processed_image_class'].iloc[i] != 'outlier' and pd.isna(df_merge['day_of_year_OCR'].iloc[i]) == False:
                        df_merge['processed_image_class'].iloc[i] = 'num2'
            else:
                df_merge = df_result
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
        for i in range(len(df_merge)):
            if df_merge['processed_image_class'].iloc[i] == 'loss':
                if df_merge['details'].iloc[i] == 'OCR read metadata contains letters':
                    for col in md_cols:
                        if col in df_merge.columns:
                            df_merge[col].iloc[i] = np.nan
            elif df_merge['processed_image_class'].iloc[i] == 'num2':
                for col in md_cols:
                    if col in df_merge.columns:
                        df_merge[col].iloc[i] = np.nan
                n_OCR_read += 1
        
        #If num2 metadata type is detected, classify images with all other metadata types as loss:
        if len(df_read) > 0:
            for i in range(len(df_merge)):
                if df_merge['processed_image_class'].iloc[i] == 'num':
                    df_merge['processed_image_class'].iloc[i] = 'loss'
                    df_merge['details'].iloc[i] = 'metadata could not be read by OCR'
                    for col in md_cols:
                        df_merge[col].iloc[i] = np.nan
                if df_merge['processed_image_class'].iloc[i] == 'dot':
                    df_merge['processed_image_class'].iloc[i] = 'loss'
                    df_merge['details'].iloc[i] = 'metadata could not be read by OCR'
                    for col in md_cols:
                        df_merge[col].iloc[i] = np.nan
        
        #If num2 metadata type is not detected:
        if len(df_read) == 0:
            n_num = len(df_merge.loc[df_merge['processed_image_class'] == 'num'])
            n_dot = len(df_merge.loc[df_merge['processed_image_class'] == 'dot'])
            #If num type metadata is the majority, classify dot type images as loss:
            if n_num > n_dot:
                for i in range(len(df_merge)):
                    if df_merge['processed_image_class'].iloc[i] == 'dot':
                        df_merge['processed_image_class'].iloc[i] = 'loss'
                        df_merge['details'].iloc[i] = 'metadata was interpreted to be dot type'
                        for col in md_cols:
                            df_merge[col].iloc[i] = np.nan
            #If dot type metadata is the majority, classify num type images as loss:
            if n_dot > n_num:
                for i in range(len(df_merge)):
                    if df_merge['processed_image_class'].iloc[i] == 'num':
                        df_merge['processed_image_class'].iloc[i] = 'loss'
                        df_merge['details'].iloc[i] = 'metadata was interpreted to be num type'
                        for col in md_cols:
                            df_merge[col].iloc[i] = np.nan   
        
        #Save:
        df_merge.to_csv(resultDir + directory + '/' + 'result_OCRpass-' + directory + '_' + subdirectory + '.csv', index=False)
        
        end = time.time()
        t = end - start
        print('Processing time for subdirectory: ' + str(round(t/60, 1)) + ' min')
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
        if os.path.exists(logDir + 'process_log_OCR.csv'):
            df_log = pd.read_csv(logDir + 'process_log_OCR.csv')
            df_update = pd.concat([df_log, df_result_], axis=0, ignore_index=True)
            df_update.to_csv(logDir + 'process_log_OCR.csv', index=False)
        else:
            if len(df_result_) > 0:
                df_result_.to_csv(logDir + 'process_log_OCR.csv', index=False)

        #Backup 'process_log' (10% of the time)
        if randrange(10) == 7:
            df_log = pd.read_csv(logDir + 'process_log_OCR.csv')
            datetime_str = datetime.now().strftime("%Y%m%d_%Hh%M")
            os.makedirs(logDir + 'backups/', exist_ok=True)
            df_log.to_csv(logDir + 'backups/' + 'process_log_OCR-' + datetime_str + '.csv', index=False)

        #Check stop conditions
        if len(subdir_ids_rem) < 2:
            print('Stop!')
            stop_condition = True
        if stop_condition_counter == stop_loop_threshold:
            print('Stop!')
            stop_condition = True