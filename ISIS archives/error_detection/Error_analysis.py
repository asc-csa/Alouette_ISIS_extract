# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 08:22:02 2023

@author: mfortier
"""

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
pd.options.mode.chained_assignment = None  # Remove Setting with copy warning in pandas

def concatResults(rootDir,resultDir):
    
    df_result = pd.DataFrame()
    i = 0
    for file in os.listdir(resultDir):
        if 'R' in file:
            directory = file
            for file2 in os.listdir(resultDir + directory + '/'):
                if 'result_scan_error_detect-' in file2:
                    fn_parts = file2.split('_')
                    subdirectory = fn_parts[1].replace('.csv', '')
                    if i > 0:
                        if i % 100 == 0:
                            df_result = pd.read_csv(resultDir + 'result_BATCH_I_raw.csv', low_memory=False)
                            print(len(df_result))
                    try:
                        df_load = pd.read_csv(resultDir + directory + '/' + file2, sep=',')
                        n = len(df_load)
                    except pd.errors.EmptyDataError:
                        n = 0
                        df_load = pd.DataFrame()
                    df_result = pd.concat([df_result, df_load])
                    i += 1
                    if i % 100 == 0:
                        print('Now saving the ' + str(i) + 'th result...')
                        df_result.to_csv(resultDir + 'result_BATCH_I_raw.csv', index=False)
    df_result.to_csv(resultDir + 'result_BATCH_I_raw.csv', index=False)
    return()

def aspectPlot(df,ylabel,title):
    
    # calculate aspect ratios
    w = df['width'].astype(int)
    h = df['height'].astype(int)
    aspect_ratios = w/h
    
    fig, ax = plt.subplots()
    
    # make histogram data more clear (larger bins and more zoomed in)
    ax.hist(aspect_ratios, bins=np.arange(0.0, 10.0, 0.20), alpha=0.8, weights=np.ones(len(df)) / len(df))
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xlabel('width/height [pixels]')
    ax.set_ylabel(ylabel)
    ax.set_xlim(0,10)
    ax.set_xticks(np.arange(0.0, 10.0, 1.0),minor=True)
    ax.grid(which='both',axis='x',alpha=0.7,ls='--')
    
    # add vertical line at w/h = 1.2
    #plt.vlines(x=1.2, ymin=0, ymax=120000, colors='purple', linestyles='--', label='width / height =1.2')
    #plt.legend()
    
    mu = aspect_ratios.mean()
    med = aspect_ratios.median()
    sigma = aspect_ratios.std()
    
    textstr = '\n'.join((
        r'$\mathrm{average}=%.2f$' % (mu, ),
        r'$\mathrm{median}=%.2f$' % (med, ),
        r'$\mathrm{sd}=%.2f$' % (sigma, )))
    
    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='white', edgecolor='lightgrey')
    
    # place a text box in upper left in axes coords
    fig.text(0.7, 0.8, textstr,va='top', bbox=props)
    fig.suptitle(title)
    
    n = len([ x for x in aspect_ratios if x < 1.2 ])
    print(round(100*n/filtered_down_len_2,2), '% of ionograms have w/h < 1.2')
    
    return(fig)

def digitPlot(df,ylabel,title):
    
    # log histogram of metadata readings
    digit_counts = df['digit_count'].astype(float)
    
    fig, ax = plt.subplots()
    
    ax.hist(digit_counts, bins=np.arange(0., 60., 5.), alpha=0.8, weights=np.ones(len(df)) / len(df))
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xlabel('total digits detected in ionogram [count]')
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(0., 60., 5.),minor=True)
    ax.grid(which='both',axis='x',alpha=0.7,ls='--')
    
    mu = digit_counts.mean()
    med = digit_counts.median()
    sigma = digit_counts.std()
    
    textstr = '\n'.join((
        r'$\mathrm{average}=%.2f$' % (mu, ),
        r'$\mathrm{median}=%.2f$' % (med, ),
        r'$\mathrm{sd}=%.2f$' % (sigma, )))
    
    props = dict(boxstyle='round', facecolor='white', edgecolor='lightgrey')
    fig.text(0.7, 0.8, textstr,va='top', bbox=props)
    fig.suptitle(title)
    
    # percentage with digit_count > 15
    n = len([ x for x in digit_counts if x > 15 ])
    print(round(100*n/filtered_down_len_2,2), '% of ionograms have digit_count > 15')

    return(fig)

def ISISinfo(df_result):
    
    # ISIS text on images
    df_isis = df_result.loc[df_result['says_isis'] == 'True']
    isis_img_pct = 100*len(df_isis)/filtered_down_len_2
    print('total number of images:', filtered_down_len_2)
    print('number of images with flagged ISIS text:', len(df_isis))
    print('percentatge of ionogram images with flagged ISIS text:', round(isis_img_pct,  2), '%\n\n')
    
    # unique ISIS subdirectories
    subdirs_isis = df_isis['Subdirectory']
    unique_subdirs_isis = set(subdirs_isis)
    subdirs_tot = df_result['Subdirectory']
    unique_subdirs_tot = set(subdirs_tot)
    isis_subdir_pct = 100*len(unique_subdirs_isis)/len(unique_subdirs_tot)
    print('total number of subdirectories:', len(unique_subdirs_tot))
    print('number of subdirectories with flagges ISIS text:', len(unique_subdirs_isis))
    print('percentatge of subdirectories with flagged ISIS text:', round(isis_subdir_pct,  2), '%\n\n')

    return()

def filterData(df):
    
    lowdigits = df.loc[(df['digit_count'].astype(int)<10)]
    lowdigits['aspect'] = lowdigits['width'].astype(int)/lowdigits['height'].astype(int)
    outofphase = lowdigits.loc[lowdigits['aspect']>=2.4]
    
    return(outofphase)


rootDir = 'L:/DATA/Alouette_I/BATCH_I_scan_error_detection_Run1/'
#processedDir = rootDir + '04_processed/'
resultDir = rootDir + '01_result/'
#logDir = rootDir + '06_log/'

concatResults(rootDir,resultDir)

df_result = pd.read_csv(resultDir + 'result_BATCH_I_raw.csv')
total_len = len(df_result)

# filter out error rows
df_result = df_result.loc[df_result['says_isis'] != 'ERR']
filtered_down_len_1 = len(df_result)
print('rows filtered out due to errors:', total_len - filtered_down_len_1)

# drop dupplicates
df_result.drop_duplicates(inplace=True)
filtered_down_len_2 = len(df_result)
print('rows filtered out due to dupplicates:', filtered_down_len_1 - filtered_down_len_2)

aspectPlot(df_result,'number of ionograms [count]','Aspect of all ionograms')
digitPlot(df_result,'number of ionograms [count]','Number of digits of all ionograms')
ISISinfo(df_result)

outofphase = filterData(df_result)
#outofphase.to_csv(resultDir + 'possibly_outofphase_batchII_raw.csv', index=False)

df_result['width']=df_result['width'].astype(int)
df_result['height']=df_result['height'].astype(int)
df_result['digit_count']=df_result['digit_count'].astype(int)

df_sub = df_result.groupby(['Directory','Subdirectory'])['width','height','digit_count'].mean()
df_sub.reset_index(inplace=True)
aspectPlot(df_sub,'percentage of subdirectories','Aspect of all ionograms')
digitPlot(df_sub,'percentage of subdirectories','Number of digits of all ionograms')
outofphase = filterData(df_sub)
df_sub.to_csv(resultDir + 'outofphase_metrics_ISIS_raw.csv')
pathList = pd.DataFrame()
pathList['Directory']=outofphase['Directory']
pathList['Subdirectory']=outofphase['Subdirectory']
pathList.drop_duplicates(inplace=True)
pathList.to_csv(resultDir + 'outofphase_subdirectories_ISIS_raw.csv')
#df_sub['out_of_phase']=False

df_problematic = pd.read_csv(resultDir + 'Report_Natalina.csv',usecols=['Directory','Subdirectory','Problem'])
df_outofphase = df_problematic.loc[df_problematic['Problem']=='out-of-phase']
outofphase_list = df_outofphase['Subdirectory'].tolist()
df_problematic.reset_index()
df_problematic.drop(list(np.where(df_problematic['Problem']=='out-of-phase')[0]),inplace=True)
other = df_problematic['Subdirectory'].tolist()

for i in range(len(df_sub)):
    if df_sub['Subdirectory'][i] in outofphase_list:
        df_sub['out_of_phase'][i]=True

outofphase = df_sub.loc[df_sub['out_of_phase']==True]
okay = df_sub.loc[df_sub['out_of_phase']==False]
okay.reset_index(inplace=True)
for i in range(len(okay)):
    if okay['Subdirectory'][i] in other:
        okay.drop(i,inplace=True)

fig1 = aspectPlot(outofphase, 'percentage of subdirectories','Aspect of out of phase subdirectories')
fig1.savefig('C:/Users/mfortier/Documents/Alouette/Presentation_images/aspect_outofphase.png')
fig2 = aspectPlot(okay, 'percentage of subdirectories','Aspect of acceptable subdirectories')
fig2.savefig('C:/Users/mfortier/Documents/Alouette/Presentation_images/aspect_acceptable.png')

fig3 = digitPlot(outofphase, 'percentage of subdirectories','Number of digits for out of phase subdirectories')
fig3.savefig('C:/Users/mfortier/Documents/Alouette/Presentation_images/digits_outofphase.png')
fig4 = digitPlot(okay, 'percentage of subdirectories','Number of digits for acceptable subdirectories')
fig4.savefig('C:/Users/mfortier/Documents/Alouette/Presentation_images/digits_acceptable.png')

plt.show()

plt.hist(outofphase['width']/outofphase['height'],bins=np.arange(0.0, 10.0, 0.20), edgecolor='black',
         alpha=0.6, weights=np.ones(len(outofphase)) / len(outofphase),label='out of phase')
plt.hist(okay['width']/okay['height'],bins=np.arange(0.0, 10.0, 0.20), edgecolor='black',
         alpha=0.6, weights=np.ones(len(okay)) / len(okay),label='okay')
plt.legend()
plt.show()

plt.hist(outofphase['digit_count'],bins=np.arange(0., 60., 5.), edgecolor='black',
         alpha=0.5, weights=np.ones(len(outofphase)) / len(outofphase),label='out of phase')
plt.hist(okay['digit_count'],bins=np.arange(0., 60., 5.), edgecolor='black',
         alpha=0.5, weights=np.ones(len(okay)) / len(okay),label='okay')
plt.legend()
plt.show()

