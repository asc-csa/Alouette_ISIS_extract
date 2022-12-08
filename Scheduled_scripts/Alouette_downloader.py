#Download Subdirectories from FTP

import pandas as pd
import os
import shutil
import time
from datetime import datetime
import ftplib
from random import randrange


process_on_VDI = False

#Set-up Directories
rootDir_local = 'C:/Users/rnaidoo/Documents/Projects_data/Alouette_I/' #files on C:/ are not persistent on VDI
downloadingDir = rootDir_local + '01_downloading/'
downloadedDir = rootDir_local + '02_downloaded/'
if process_on_VDI:
    logDir = '//scientific/L-MP-Data/Massive files/Python/rnaidoo/Alouette_I/' #DO NOT CHANGE
else:
    logDir = rootDir_local + '05_result/'
   
    
# Connect to FTP Server
HOSTNAME = "donnees-data.asc-csa.gc.ca"
USERNAME = "Anonymous"
PASSWORD = ""
ftp = ftplib.FTP(HOSTNAME, USERNAME, PASSWORD)
print('Connected to ftp server: ' + HOSTNAME)
ftp_rootpath = '/users/OpenData_DonneesOuvertes/pub/AlouetteData/Alouette Data/'


#Functions
def move_images(old_dir, new_dir, roll, subdir):
    oldDir = old_dir + roll + '/' + subdir + '/'
    newDir = new_dir + roll + '/' + subdir + '/'
    os.makedirs(newDir, exist_ok=True)
    for file in os.listdir(oldDir):
        os.rename(oldDir+file, newDir+file)
    shutil.rmtree(old_dir + roll + '/')


def draw_random_subdir(roll, subdir_list, logDir):
    
    subdirectory = subdir_list[randrange(len(subdir_list))]
    
    #Check randomly-selected roll and subdirectory against the 'download_log'
    if os.path.exists(logDir + 'download_log.csv'):
        df_log = pd.read_csv(logDir + 'download_log.csv')
        df_search = df_log.loc[(df_log['Roll'] == roll) & (df_log['Subdirectory'] == subdirectory)]
        if len(df_search) > 0:
            print(roll + '/' + subdirectory + ' already downloaded!')
            return ''
        else:
            return subdirectory
    else:
        return subdirectory


#Download a random subdirectory by FTP
ftp.cwd(ftp_rootpath)
roll_list = ftp.nlst()
roll = roll_list[randrange(len(roll_list))]
ftp.cwd(roll)
subdir_list = ftp.nlst()
subdirectory = ''
while (subdirectory == ''):
    subdirectory = draw_random_subdir(roll=roll, subdir_list=subdir_list, logDir=logDir)

saveDir = downloadingDir + roll + '/' + subdirectory + '/'
os.makedirs(saveDir, exist_ok=True)
ftp.cwd(ftp_rootpath + '/' + roll + '/' + subdirectory + '/')

start = time.time()
n_dl = len(ftp.nlst())
print('Downloading ' + roll + '/' + subdirectory + '/ subdirectory ('  + str(n_dl) + ' images)')
for file in ftp.nlst():
    local_file = open(saveDir + file, "wb")
    ftp.retrbinary("RETR " + file, local_file.write)
    local_file.close()
    #print('Downloaded: ' + file)
end = time.time()
t = end - start
print('Download time for subdirectory: ' + str(round(t/60, 1)) + ' min')


#Record subdirectory name in download log
df_result = pd.DataFrame({
    'Roll': roll,
    'Subdirectory': subdirectory,
    'Images_downloaded': n_dl,
    'Download_time': t,
    'Download_timestamp': datetime.fromtimestamp(end)
}, index=[0])
if os.path.exists(logDir + 'download_log.csv'):
    df_log = pd.read_csv(logDir + 'download_log.csv')
    df_update = pd.concat([df_log, df_result], axis=0, ignore_index=True)
    df_update.to_csv(logDir + 'download_log.csv', index=False)
else:
    df_result.to_csv(logDir + 'download_log.csv', index=False)


#Backup 'download_log' (10% of the time):
if randrange(10) == 7:
    df_log = pd.read_csv(logDir + 'download_log.csv')
    datetime_str = datetime.now().strftime("%Y%m%d_%Hh%M")
    os.makedirs(logDir + 'backups/', exist_ok=True)
    df_log.to_csv(logDir + 'backups/' + 'download_log-' + datetime_str + '.csv', index=False)


#Move fully downloaded subdirectory to 02_processing folder:
move_images(old_dir=downloadingDir, new_dir=downloadedDir, roll=roll, subdir=subdirectory)