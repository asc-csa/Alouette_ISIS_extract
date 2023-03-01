#Process Subdirectories

import sys
import pandas as pd
import os
import subprocess
import shutil
import time
from datetime import datetime
from random import randrange

import warnings
warnings.filterwarnings('ignore')

#Set parameters
instance = sys.argv[1]
user = 'Rav Super' + instance #e.g: 'Rav Super2'
process_on_VDI = True

#Set directories
rootDir_local = 'C:/Users/rnaidoo/Documents/Projects_data/Alouette_I/SuperVDI' + instance + '/BATCH_I_Run1_rp/'
rootDir_L_ = 'L:/DATA/Alouette_I/BATCH_I_Run1/'
rootDir_L = 'L:/DATA/Alouette_I/BATCH_I_Run1_rp/'
downloadedDir = rootDir_local + '02_downloaded/'
processingDir = rootDir_local + '03_processing/'
result_localDir = rootDir_local + '05a_result_local/'
if process_on_VDI:
    processedDir = rootDir_L_ + '04_processed/' 
    unprocessedDir = rootDir_L_ + '04a_unprocessed/'
    resultDir = rootDir_L + '05_result/' 
    logDir = rootDir_L + '06_log/'
    move_to_L = True
else:
    processedDir = rootDir_local + '04_processed/' 
    unprocessedDir = rootDir_local + '04a_unprocessed/' 
    resultDir = rootDir_local + '05_result/' 
    logDir = rootDir_local + '06_log/'
    move_to_L = False
    


#Functions
def move_images(old_dir, new_dir, roll, subdir, copy_to_other_drive=False, delete_old_dir=False):
    oldDir = old_dir + roll + '/' + subdir + '/'
    newDir = new_dir + roll + '/' + subdir + '/'
    os.makedirs(newDir, exist_ok=True)
    
    if copy_to_other_drive:
        if os.path.exists(oldDir):
            for file in os.listdir(oldDir):
                shutil.copyfile(oldDir+file, newDir+file)
    else:
        if os.path.exists(oldDir):
            for file in os.listdir(oldDir):
                os.rename(oldDir+file, newDir+file)
    
    if delete_old_dir:
        if os.path.exists(oldDir):
            shutil.rmtree(old_dir + roll + '/' + subdir + '/')
            if len(os.listdir(old_dir + roll + '/')) == 0:
                shutil.rmtree(old_dir + roll + '/')
                


#Re-process list of subdirectories
df_reprocess = pd.read_csv(logDir + 'image_inventory.csv') #reprocess_list.csv
df_reprocess = df_reprocess.sample(frac=1)
print(len(df_reprocess))
reprocess_list = df_reprocess['subdir_id']

for subdir in reprocess_list:
    
    start = time.time()
    
    subdir_id_parts = subdir.split('_')
    roll = subdir_id_parts[0]
    subdirectory = subdir_id_parts[1]
    subdir_path_end = roll + '/' + subdirectory + '/'
    
    #Clear any old subdirectories in processingDir
    for file in os.listdir(processingDir):
        if 'R' in file:
            shutil.rmtree(processingDir + file)
    
    #Clear intermediate results in result_localDir
    for file in os.listdir(result_localDir):
        if 'df' in file:
            os.remove(result_localDir + file)
        else:
            shutil.rmtree(result_localDir + file)
    
    #Retrieve subdirectory
    if os.path.exists(processedDir + subdir_path_end):
        move_images(old_dir=processedDir, new_dir=processingDir, roll=roll, subdir=subdirectory, copy_to_other_drive=True)
    elif os.path.exists(unprocessedDir + subdir_path_end):
        move_images(old_dir=unprocessedDir, new_dir=processingDir, roll=roll, subdir=subdirectory, copy_to_other_drive=True)
    else:
        print('Cannot find subdirectory ' + subdir + '!')
        continue
    
    #Process
    print('')
    print('Processing ' + subdir_path_end + ' subdirectory...')
    subprocess.run('./scan2data/user_input.py' + ' ' + processingDir + ' ' + result_localDir, shell=True)

    #Consolidate results
    if os.path.exists(result_localDir + 'df_dot.csv'):
        df_dot = pd.read_csv(result_localDir + 'df_dot.csv')
        n_dot = len(df_dot)
        df_dot['processed_image_class'] = 'dot'
        os.remove(result_localDir + 'df_dot.csv')
    else:
        df_dot = pd.DataFrame()
        n_dot = 0

    if os.path.exists(result_localDir + 'df_num.csv'):
        df_num = pd.read_csv(result_localDir + 'df_num.csv')
        n_num = len(df_num)
        df_num['processed_image_class'] = 'num'
        os.remove(result_localDir + 'df_num.csv')
    else:
        df_num = pd.DataFrame()
        n_num = 0

    if os.path.exists(result_localDir + 'df_loss.csv'):
        df_loss = pd.read_csv(result_localDir + 'df_loss.csv')
        n_loss = len(df_loss)
        df_loss['processed_image_class'] = 'loss'
        os.remove(result_localDir + 'df_loss.csv')
    else:
        df_loss = pd.DataFrame()
        n_loss = 0

    if os.path.exists(result_localDir + 'df_outlier.csv'):
        df_outlier = pd.read_csv(result_localDir + 'df_outlier.csv')
        n_outlier = len(df_outlier)
        df_outlier['processed_image_class'] = 'outlier'
        os.remove(result_localDir + 'df_outlier.csv')
    else:
        df_outlier = pd.DataFrame()
        n_outlier = 0

    df_tot = pd.concat([df_dot, df_num, df_loss, df_outlier])
    if len(df_tot) > 0:
        df_tot['Roll'] = roll
        df_tot['Subdirectory'] = subdirectory
        if 'file_name' in df_tot.columns:
            df_tot['filename'] = df_tot['file_name'].str.replace(processingDir + roll + '/' + subdirectory, '')
            df_tot['filename'] = df_tot['filename'].str.replace('\\', '')
            df_tot['filename'] = df_tot['filename'].str.replace('/', '')
        else:
            df_tot['filename'] = 'unknown'
        df_tot = df_tot.drop(columns=['file_name', 'mapped_coord', 'subdir_name', 'raw', 'ionogram', 'raw_metadata', 
                                      'trimmed_metadata', 'padded', 'dilated_metadata'], errors='ignore')
    os.makedirs(resultDir + roll + '/', exist_ok=True)
    df_tot.to_csv(resultDir + roll + '/' + 'result-' + roll + '_' + subdirectory + '.csv', index=False)

    #move mapped_coords to '05_result'
    mapped_coords_localDir = result_localDir + 'mapped_coords/'
    mapped_coordsDir = resultDir + 'mapped_coords/'
    move_images(old_dir=mapped_coords_localDir, new_dir=mapped_coordsDir, roll=roll, subdir=subdirectory, copy_to_other_drive=move_to_L)
    
    end = time.time()
    t = end - start
    print('Processing time for subdirectory: ' + str(round(t/60, 1)) + ' min')
    print('')

    #Record performance
    n_processed = n_dot + n_num + n_loss + n_outlier
    df_result_ = pd.DataFrame({
        'Roll': roll,
        'Subdirectory': subdirectory,
        'Images_processed': n_processed,
        'Images_dot': n_dot,
        'Images_num': n_num,
        'Images_loss': n_loss,
        'Images_outlier': n_outlier,
        'Process_time': t,
        'Process_timestamp': datetime.fromtimestamp(end),
        'User': user,
        'subdir_id': roll + '_' + subdirectory
    }, index=[0])
    if os.path.exists(logDir + 'process_log.csv'):
        df_log = pd.read_csv(logDir + 'process_log.csv')
        df_update = pd.concat([df_log, df_result_], axis=0, ignore_index=True)
        df_update.to_csv(logDir + 'process_log.csv', index=False)
    else:
        if len(df_result_) > 0:
            df_result_.to_csv(logDir + 'process_log.csv', index=False)

    #Backup 'process_log' (10% of the time)
    if randrange(10) == 7:
        df_log = pd.read_csv(logDir + 'process_log.csv')
        datetime_str = datetime.now().strftime("%Y%m%d_%Hh%M")
        os.makedirs(logDir + 'backups/', exist_ok=True)
        df_log.to_csv(logDir + 'backups/' + 'process_log-' + datetime_str + '.csv', index=False)

    #Move to '04_processed' or '04a_unprocessed'
    if n_processed > 0:
        move_images(old_dir=processingDir, new_dir=processedDir, roll=roll, subdir=subdirectory, copy_to_other_drive=move_to_L, delete_old_dir=True)
    else:
        move_images(old_dir=processingDir, new_dir=unprocessedDir, roll=roll, subdir=subdirectory, copy_to_other_drive=move_to_L, delete_old_dir=True)
    