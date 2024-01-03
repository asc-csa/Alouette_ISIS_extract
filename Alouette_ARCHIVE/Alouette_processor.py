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
user_prefix = sys.argv[4]
instance = sys.argv[5]
user = user_prefix + instance #e.g: 'Rav Super2'
process_on_VDI = True
stop_loop_threshold = 2640 #max while loops to prevent infinite loop

#Set directories
rootDir_local = sys.argv[3] + instance + '/Stage_1/'
dataDir_L = sys.argv[1]
rootDir_L = sys.argv[2]
downloadedDir = rootDir_local + '02_downloaded/'
processingDir = rootDir_local + '03_processing/'
result_localDir = rootDir_local + '05a_result_local/'
if process_on_VDI:
    processedDir = rootDir_L + '04_processed/' 
    unprocessedDir = rootDir_L + '04a_unprocessed/'
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
def move_images(old_dir, new_dir, directory, subdir, copy_to_other_drive=False, delete_old_dir=False):
    oldDir = old_dir + directory + '/' + subdir + '/'
    newDir = new_dir + directory + '/' + subdir + '/'
    os.makedirs(newDir, exist_ok=True)

    if copy_to_other_drive:
        if os.path.exists(oldDir):
            for file in os.listdir(oldDir):
                shutil.copyfile(oldDir+file, newDir+file)
    elif os.path.exists(oldDir):
        for file in os.listdir(oldDir):
            os.rename(oldDir+file, newDir+file)

    if delete_old_dir and os.path.exists(oldDir):
        shutil.rmtree(old_dir + directory + '/' + subdir + '/')
        if len(os.listdir(old_dir + directory + '/')) == 0:
            shutil.rmtree(old_dir + directory + '/')
                

def draw_random_subdir(subdir_ids_list, logDir):
    
    subdir_id = subdir_ids_list[randrange(len(subdir_ids_list))]
    subdir_id_parts = subdir_id.split('_')
    directory = subdir_id_parts[0]
    subdirectory = subdir_id_parts[1]

    if not os.path.exists(logDir + 'process_log.csv'):
        return directory, subdirectory
    df_log = pd.read_csv(logDir + 'process_log.csv')
    df_search = df_log.loc[(df_log['Directory'] == directory) & (df_log['Subdirectory'] == subdirectory)]
    if len(df_search) <= 0:
        return directory, subdirectory
    print(directory + '/' + subdirectory + ' already processed!')
    return ''
    


#Process remaining subdirectories with while loop
stop_condition = False
stop_condition_counter = 0

while not stop_condition:
    start = time.time()

    #Draw random, yet to be processed subdirectory, to process
    df_inventory = pd.read_csv(logDir + 'image_inventory.csv')
    subdir_ids_tot = df_inventory['subdir_id'].unique()
    if os.path.exists(logDir + 'process_log.csv'):
        df_log = pd.read_csv(logDir + 'process_log.csv')
        subdir_ids_proc = df_log['subdir_id'].unique()
    else:
        subdir_ids_proc = []
    subdir_ids_rem = list(set(subdir_ids_tot) - set(subdir_ids_proc))
    directory, subdirectory = draw_random_subdir(subdir_ids_list=subdir_ids_rem, logDir=logDir)
    subdir_path_end = directory + '/' + subdirectory + '/'

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
    if os.path.exists(dataDir_L + subdir_path_end):
        move_images(old_dir=dataDir_L, new_dir=processingDir, directory=directory, subdir=subdirectory, copy_to_other_drive=True)
    #elif os.path.exists(unprocessedDir + subdir_path_end):
    #    move_images(old_dir=unprocessedDir, new_dir=processingDir, directory=directory, subdir=subdirectory, copy_to_other_drive=True)
    else:
        print('Cannot find subdirectory ' + subdirectory + '!')
        continue

    #Process
    print('')
    print('Processing ' + subdir_path_end + ' subdirectory...')
    print(f'{len(subdir_ids_rem)} subdirectories to go!')
    subprocess.run('python user_input.py' + ' ' + processingDir + ' ' + result_localDir, shell=True, cwd='scan2data/')

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
        df_tot['Directory'] = directory
        df_tot['Subdirectory'] = subdirectory
        if 'file_name' in df_tot.columns:
            df_tot['filename'] = df_tot['file_name'].str.replace(processingDir + directory + '/' + subdirectory, '')
            df_tot['filename'] = df_tot['filename'].str.replace('\\', '')
            df_tot['filename'] = df_tot['filename'].str.replace('/', '')
        else:
            df_tot['filename'] = 'unknown'
        df_tot = df_tot.drop(columns=['file_name', 'mapped_coord', 'subdir_name', 'raw', 'ionogram', 'raw_metadata', 
                                      'trimmed_metadata', 'padded', 'dilated_metadata'], errors='ignore')
    os.makedirs(resultDir + directory + '/', exist_ok=True)
    df_tot.to_csv(resultDir + directory + '/' + 'result-' + directory + '_' + subdirectory + '.csv', index=False)

    #move mapped_coords to '05_result'
    mapped_coords_localDir = result_localDir + 'mapped_coords/'
    mapped_coordsDir = resultDir + 'mapped_coords/'
    move_images(old_dir=mapped_coords_localDir, new_dir=mapped_coordsDir, directory=directory, subdir=subdirectory, copy_to_other_drive=move_to_L)

    end = time.time()
    t = end - start
    print(f'Processing time for subdirectory: {str(round(t / 60, 1))} min')
    print('')

    #Record performance
    n_processed = n_dot + n_num + n_loss + n_outlier
    df_result_ = pd.DataFrame({
        'Directory': directory,
        'Subdirectory': subdirectory,
        'Images_processed': n_processed,
        'Images_dot': n_dot,
        'Images_num': n_num,
        'Images_loss': n_loss,
        'Images_outlier': n_outlier,
        'Process_time': t,
        'Process_timestamp': datetime.fromtimestamp(end),
        'User': user,
        'subdir_id': directory + '_' + subdirectory
    }, index=[0])
    if os.path.exists(logDir + 'process_log.csv'):
        df_log = pd.read_csv(logDir + 'process_log.csv')
        df_update = pd.concat([df_log, df_result_], axis=0, ignore_index=True)
        df_update.to_csv(logDir + 'process_log.csv', index=False)
    elif len(df_result_) > 0:
        df_result_.to_csv(logDir + 'process_log.csv', index=False)

    #Backup 'process_log' (10% of the time)
    if randrange(10) == 7:
        df_log = pd.read_csv(logDir + 'process_log.csv')
        datetime_str = datetime.now().strftime("%Y%m%d_%Hh%M")
        os.makedirs(logDir + 'backups/', exist_ok=True)
        df_log.to_csv(logDir + 'backups/' + 'process_log-' + datetime_str + '.csv', index=False)

    #Move to '04_processed' or '04a_unprocessed'
    if n_processed > 0:
        move_images(old_dir=processingDir, new_dir=processedDir, directory=directory, subdir=subdirectory, copy_to_other_drive=move_to_L, delete_old_dir=True)
    else:
        move_images(old_dir=processingDir, new_dir=unprocessedDir, directory=directory, subdir=subdirectory, copy_to_other_drive=move_to_L, delete_old_dir=True)

    #Check stop conditions
    if len(subdir_ids_rem) < 2:
        print('Stop!')
        stop_condition = True
    if stop_condition_counter == stop_loop_threshold:
        print('Stop!')
        stop_condition = True