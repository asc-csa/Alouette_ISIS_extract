#Once the dataframe containing all the extracted traces has been saved to pickle, it can be loaded here and the traces can be analyzed

import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter
import math
from itertools import compress

from filtering import resample, custom_filter
from conversions import convert_pixel_to_mapping, convert_mapping_to_pixel
from utils import find_nearest, make_plot, plot_ionogram_image, get_max_depth

if __name__ == '__main__':
    #Load pickle file containing data for subdirectory
    file_name = "R014207978F_314.pkl"
    file_path = os.path.join(os.path.pardir, 'pickle', file_name)
    df_processed = pd.read_pickle(file_path)

    #Save directory
    save_path = os.path.join(os.path.pardir, 'saved_results')

    #Take a subset of the dataset for testing
    # df_processed = df_processed.iloc[0:5]

    #Load the mapping_Hz and mapping_km for the subdirectory
    mapping_Hz = df_processed.iloc[0]['mapping_Hz']
    mapping_km = df_processed.iloc[0]['mapping_km']

    #Dataframe containing the trace coordinates
    df_coord = df_processed[['file_name', 'mapped_coord', 'raw_coord', 'window_coord', 'limits']]


    #Threshold the km value to eliminate noise
    km_threshold = 200
    # 1) Get raw coord
    # 2) Convert raw coord into km, Hz
    # 3) Filter the trace with rolling mean, SG
    # 4) Save filtered trace and the parameters extracted
    # 5) Convert the trace back to pixels for plotting on top of the ionogram

    for index, row in df_coord.iterrows():
        #Ionogram title
        title = ' '.join(row['file_name'].split('\\')[-3:])

        #Extract x and y for each trace
        mapped_coord = row['mapped_coord']
        raw_coord = row['raw_coord']
        x = raw_coord[:,0]
        y = raw_coord[:,1]

        #Convert pixel values to Hz, km
        x, y = convert_pixel_to_mapping(x, y, mapping_Hz, mapping_km)

        x = x[ y > km_threshold ]
        y = y[ y > km_threshold ]

        sorted_xy = sorted(zip(x,y))
        x, y = zip(*sorted_xy)

        #Resample the x, y
        x, y = resample(x, y, delta_x = 0.01)
        x_pixel, y_pixel = convert_mapping_to_pixel(x, y, mapping_Hz, mapping_km)  # Raw pixel values after thresholding and resampling

        #Get the max depth of the trace in Hz, km. Then, convert to pixel coordinates
        x_max_depth, y_max_depth = get_max_depth(x,y)
        x_max_depth_pixel, y_max_depth_pixel = convert_mapping_to_pixel(x_max_depth, y_max_depth, mapping_Hz, mapping_km)


        #Plot raw_coord against the new x_pixel, y_pixel after thresholding and resampling
        plt.figure()
        plt.scatter(raw_coord[:,0], raw_coord[:,1], s=1, color='blue')
        plt.scatter(x_pixel, y_pixel, s=5, color='red')
        plt.scatter(x_max_depth_pixel, y_max_depth_pixel, s=100, color='pink')
        plot_ionogram_image(row['file_name'], row['limits'], title, ' original raw(blue) vs resampled raw(red)')
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.savefig( os.path.join(save_path, title))
        plt.close()



