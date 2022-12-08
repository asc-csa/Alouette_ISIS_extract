#Process Subdirectories

import pandas as pd
import os
import shutil
import time
from datetime import datetime
from random import randrange

import warnings
warnings.filterwarnings('ignore')


process_on_VDI = False

#Set-up Directories
rootDir_local = 'C:/Users/rnaidoo/Documents/Projects_data/Alouette_I/' #C: is not persistent on VDI
rootDir_U = 'U:/Data_Science/Projects_data/Alouette_I/'
downloadedDir = rootDir_local + '02_downloaded/'
processingDir = rootDir_local + '03_processing/'
if process_on_VDI:
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
