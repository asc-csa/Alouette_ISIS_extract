#Download Subdirectories from FTP

import sys
import pandas as pd
import os
import shutil
import time
from datetime import datetime
import ftplib
from random import randrange


#Set parameters
user = sys.argv[3]
wait = 4 #in minutes
stop_loop_threshold = 6000 #max while loops to prevent infinite loop

#Set directories
rootDir = sys.argv[1] #The file path to the root directory for your project 
downloadingDir = rootDir + '01_downloading/'
downloadedDir = rootDir + '02_downloaded/'
logDir = rootDir + '06_log/'



#Functions
def move_images(old_dir, new_dir, directory, subdirectory, copy_to_other_drive=False):
    oldDir = old_dir + directory + '/' + subdirectory + '/'
    newDir = new_dir + directory + '/' + subdirectory + '/'
    os.makedirs(newDir, exist_ok=True)
    
    if copy_to_other_drive:
        for file in os.listdir(oldDir):
            shutil.copyfile(oldDir+file, newDir+file)
    else:
        for file in os.listdir(oldDir):
            os.rename(oldDir+file, newDir+file)
    
    shutil.rmtree(old_dir + directory + '/' + subdirectory + '/')
    if len(os.listdir(old_dir + directory + '/')) == 0:
        shutil.rmtree(old_dir + directory + '/')


def draw_random_subdir2(subdir_ids_list, logDir):
    
    subdir_id = subdir_ids_list[randrange(len(subdir_ids_list))]
    subdir_id_parts = subdir_id.split('_')
    directory = subdir_id_parts[0]
    subdirectory = subdir_id_parts[1]
    
    #Check randomly-selected directory and subdirectory against the 'download_log'
    if os.path.exists(logDir + 'download_log.csv'):
        df_log = pd.read_csv(logDir + 'download_log.csv')
        df_search = df_log.loc[(df_log['Directory'] == directory) & (df_log['Subdirectory'] == subdirectory)]
        if len(df_search) > 0:
            print(directory + '/' + subdirectory + ' already downloaded!')
            return ''
        else:
            return directory, subdirectory
    else:
        return directory, subdirectory
    


#Check if subdirectory needs to be downloaded, then download random subdirectory
stop_condition = False
stop_condition_counter = 0

while stop_condition == False:
    #Download a random subdirectory if '02_downloaded' is empty
    if len(os.listdir(downloadedDir)) == 0:
        # Connect FTP Server
        HOSTNAME = "data.asc-csa.gc.ca"
        USERNAME = "anonymous"
        PASSWORD = sys.argv[2]
        ftp = ftplib.FTP(HOSTNAME, USERNAME, PASSWORD)
        print('Connected to ftp server: ' + HOSTNAME)
        ftp_rootpath = 'users/OpenData_DonneesOuvertes/pub/Alouette-ISIS/Alouette-1/'

        #Randomly draw directory and subdirectory (using draw_random_subdir2())
        df_inventory = pd.read_csv(logDir + 'image_inventory.csv')
        subdir_ids_tot = df_inventory['subdir_id'].unique()
        if os.path.exists(logDir + 'download_log.csv'):
            df_log = pd.read_csv(logDir + 'download_log.csv')
            subdir_ids_dl = df_log['subdir_id'].unique()
        else:
            subdir_ids_dl = []
        subdir_ids_rem = list(set(subdir_ids_tot) - set(subdir_ids_dl))
        directory, subdirectory = draw_random_subdir2(subdir_ids_list=subdir_ids_rem, logDir=logDir)

        #Set directories
        saveDir = downloadingDir + directory + '/' + subdirectory + '/'
        os.makedirs(saveDir, exist_ok=True)
        ftp.cwd(ftp_rootpath + '/' + directory + '/' + subdirectory + '/')

        start = time.time()
        n_dl = len(ftp.nlst())
        print('')
        print('Downloading ' + directory + '/' + subdirectory + '/ subdirectory ('  + str(n_dl) + ' images, ' + str(len(subdir_ids_rem)-1) + ' subdirectories to go)')
        for file in ftp.nlst():
            local_file = open(saveDir + file, "wb")
            ftp.retrbinary("RETR " + file, local_file.write)
            local_file.close()
            #print('Downloaded: ' + file)
        end = time.time()
        t = end - start
        print('Download time for subdirectory: ' + str(round(t/60, 1)) + ' min')
        print('')

        #Record subdirectory name in download_log
        df_result = pd.DataFrame({
            'Directory': directory,
            'Subdirectory': subdirectory,
            'Images_downloaded': n_dl,
            'Download_time': t,
            'Download_timestamp': datetime.fromtimestamp(end),
            'User': user,
            'subdir_id': directory + '_' + subdirectory
        }, index=[0])
        if os.path.exists(logDir + 'download_log.csv'):
            df_log = pd.read_csv(logDir + 'download_log.csv')
            df_update = pd.concat([df_log, df_result], axis=0, ignore_index=True)
            df_update.to_csv(logDir + 'download_log.csv', index=False)
        else:
            df_result.to_csv(logDir + 'download_log.csv', index=False)

        #Backup 'download_log' (10% of the time)
        if randrange(10) == 7:
            df_log = pd.read_csv(logDir + 'download_log.csv')
            datetime_str = datetime.now().strftime("%Y%m%d_%Hh%M")
            os.makedirs(logDir + 'backups/', exist_ok=True)
            df_log.to_csv(logDir + 'backups/' + 'download_log-' + datetime_str + '.csv', index=False)

        #Move fully downloaded subdirectory to '02_processing' folder
        move_images(old_dir=downloadingDir, new_dir=downloadedDir, directory=directory, subdir=subdirectory)
        
        stop_condition_counter += 1
    
    else:
        #Wait
        print('Wait ' + str(wait) + ' min')
        subdir_ids_rem = []
        time.sleep(wait*60)
    
    
    #Check stop conditions
    if len(subdir_ids_rem) == 1:
        print('Stop!')
        stop_condition = True
    if stop_condition_counter == stop_loop_threshold:
        print('Stop!')
        stop_condition = True

