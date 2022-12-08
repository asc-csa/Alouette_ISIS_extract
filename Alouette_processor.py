#Process Subdirectories

import sys
import pandas as pd
import os
import shutil
import time
from datetime import datetime
from random import randrange
#from scan2data import user_input
import subprocess

import warnings
warnings.filterwarnings('ignore')


process_from = sys.argv[1]


#Set-up Directories
codeDir = sys.argv[2]
rootDir_local = 'C:/Users/rnaidoo/Documents/Projects_data/Alouette_I/' #C: is not persistent on VDI
rootDir_U = 'U:/Data_Science/Projects_data/Alouette_I/'
downloadedDir = rootDir_local + '02_downloaded/'
processingDir = rootDir_local + '03_processing/'
if process_from == 'VDI':
    processedDir = rootDir_U + '04_processed/' 
    resultDir = rootDir_U + '05_result/' 
    logDir = '//scientific/L-MP-Data/Massive files/Python/rnaidoo/Alouette_I/' #DO NOT CHANGE
    move_to_U = True
else:
    processedDir = rootDir_local + '04_processed/' 
    resultDir = rootDir_local + '05_result/' 
    logDir = resultDir
    move_to_U = False


#Functions
def move_images(old_dir, new_dir, roll, subdir, copy_to_other_drive=False):
    oldDir = old_dir + roll + '/' + subdir + '/'
    newDir = new_dir + roll + '/' + subdir + '/'
    os.makedirs(newDir, exist_ok=True)
    
    if copy_to_other_drive:
        for file in os.listdir(oldDir):
            shutil.copyfile(oldDir+file, newDir+file)
    else:
        for file in os.listdir(oldDir):
            os.rename(oldDir+file, newDir+file)
    
    shutil.rmtree(old_dir + roll + '/')


#Process new fully downloaded subdirectories
df_result = pd.DataFrame()
for roll in os.listdir(downloadedDir):
    if 'R' in roll:
        for subdirectory in os.listdir(downloadedDir + roll):
            start = time.time()
            subdir_path_end = roll + '/' + subdirectory + '/'
            
            #Move to '03_processing'
            move_images(old_dir=downloadedDir, new_dir=processingDir, roll=roll, subdir=subdirectory)
            
            #Process
            print('Processing ' + subdir_path_end + ' subdirectory...')
            subprocess.run(['python', codeDir + 'scan2data/user_input.py', processingDir, resultDir])

            #Consolidate results
            if os.path.exists(resultDir + 'df_dot.csv'):
                df_dot = pd.read_csv(resultDir + 'df_dot.csv')
                n_dot = len(df_dot)
                df_dot['processed_image_class'] = 'dot'
                os.remove(resultDir + 'df_dot.csv')
            else:
                df_dot = pd.DataFrame()
                n_dot = 0
                
            if os.path.exists(resultDir + 'df_num.csv'):
                df_num = pd.read_csv(resultDir + 'df_num.csv')
                n_num = len(df_num)
                df_num['processed_image_class'] = 'num'
                os.remove(resultDir + 'df_num.csv')
            else:
                df_num = pd.DataFrame()
                n_num = 0
                
            if os.path.exists(resultDir + 'df_loss.csv'):
                df_loss = pd.read_csv(resultDir + 'df_loss.csv')
                n_loss = len(df_loss)
                df_loss['processed_image_class'] = 'loss'
                os.remove(resultDir + 'df_loss.csv')
            else:
                df_loss = pd.DataFrame()
                n_loss = 0
                
            if os.path.exists(resultDir + 'df_outlier.csv'):
                df_outlier = pd.read_csv(resultDir + 'df_outlier.csv')
                n_outlier = len(df_outlier)
                df_outlier['processed_image_class'] = 'outlier'
                os.remove(resultDir + 'df_outlier.csv')
            else:
                df_outlier = pd.DataFrame()
                n_outlier = 0
            
            df_tot = pd.concat([df_dot, df_num, df_loss, df_outlier])
            df_tot['Roll'] = roll
            df_tot['Subdirectory'] = subdirectory
            df_tot = df_tot.drop(columns=['file_name', 'mapped_coord', 'subdir_name', 'raw', 'ionogram', 'raw_metadata', 
                                          'trimmed_metadata', 'padded', 'dilated_metadata'], errors='ignore')
            df_tot.to_csv(resultDir + 'result-' + roll + '_' + subdirectory + '.csv', index=False)
            
            end = time.time()
            t = end - start
            print('Processing time for subdirectory: ' + str(round(t/60, 1)) + ' min')
            
            #Record performance
            df_result_ = pd.DataFrame({
                'Roll': roll,
                'Subdirectory': subdirectory,
                'Images_processed': n_dot + n_num + n_loss + n_outlier,
                'Images_dot': n_dot,
                'Images_num': n_num,
                'Images_loss': n_loss,
                'Images_outlier': n_outlier,
                'Process_time': t,
                'Process_timestamp': datetime.fromtimestamp(end)
            }, index=[0])
            df_result = pd.concat([df_result, df_result_], axis=0, ignore_index=True)  
            
            #Move to '04_processed'
            print("Moving images to '04_processed'")
            move_images(old_dir=processingDir, new_dir=processedDir, roll=roll, subdir=subdirectory, copy_to_other_drive=move_to_U)
            

#Add results to 'process_log'
if os.path.exists(logDir + 'process_log.csv'):
    df_log = pd.read_csv(logDir + 'process_log.csv')
    df_update = pd.concat([df_log, df_result], axis=0, ignore_index=True)
    df_update.to_csv(logDir + 'process_log.csv', index=False)
else:
    if len(df_result) > 0:
        df_result.to_csv(logDir + 'process_log.csv', index=False)


#Backup 'process_log' (10% of the time):
if randrange(10) == 7:
    df_log = pd.read_csv(logDir + 'process_log.csv')
    datetime_str = datetime.now().strftime("%Y%m%d_%Hh%M")
    os.makedirs(logDir + 'backups/', exist_ok=True)
    df_log.to_csv(logDir + 'backups/' + 'process_log-' + datetime_str + '.csv', index=False)
