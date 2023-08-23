# -*- coding: utf-8 -*-
"""
Starting code to measure and visualize the noise in an ionogram using various assesment methods

"""

# Library imports
import math
import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from skimage.restoration import estimate_sigma

sys.path.append('../')
from image_segmentation.extract_ionogram_from_scan import extract_ionogram

def noise_assesment(ionogram,
                    kernel_size= 5):
    
    """Returns the outputs of various noise assesments, including standard deviation of difference with median filtered,fast noise variance estimation, wavelet estimated noise
    
    :param ionogram: 2-dimmensional UTF-8 grayscale array of values ranging from [0,255] representing an ionogram
    :type ionogram: class: `numpy.ndarray`
    :param kernel_size: kernel size for median filtering operation, defaults to 5
    :type kernel_size: int, optional
    :returns: noise_std, noise_FNVE, noise_wavelet i.e. standard deviation of difference between raw ionogram and median filtered ionogram, noise estimated using the method outlined in J. Immerkær's Fast Noise Variance Estimation,noise estimated using the method outlined in D. L. Donoho and I. M. Johnstone's “Ideal spatial adaptation by wavelet shrinkage.”  
    :rtype: float, float, float
    """
    
    raw_iono = np.copy(ionogram)
    h,w = raw_iono.shape
    
    # Standard deviation of difference between raw ionogram and median filtered ionogram
    median_filtered_iono = cv2.medianBlur(ionogram ,kernel_size)
    noise_std = np.std(raw_iono-median_filtered_iono)
    
    
    #  J. Immerkær, “Fast Noise Variance Estimation”, 
    filter_FNVE =  [[1, -2, 1],
                    [-2, 4, -2],
                    [1, -2, 1]]
   
    noise_FNVE = np.sum(np.absolute(signal.convolve2d(raw_iono,filter_FNVE)))
    noise_FNVE = noise_FNVE * math.sqrt(0.5*math.pi)/(6*(h-2)*(w-2))
    
    # D. L. Donoho and I. M. Johnstone. “Ideal spatial adaptation by wavelet shrinkage.” 
    noise_wavelet = estimate_sigma(ionogram,average_sigmas=True)
    

    return noise_std, noise_FNVE, noise_wavelet
    

def plot_noise(list_img_path,labels):
    """Visualize the non-spectral noise of the labelled images in list_img
    
    :param list_img_path: list of paths of labelled images
    :type list_img_path: list
    :param labels: list of whether each image in list_img_path is clean (0) or noisy(1)
    :type labels: list
    
    """
    noise_array=[]
    
    for img_path in list_img_path:
        img = cv2.imread(img_path,0)
        _,iono = extract_ionogram(img)
        
        # Noise estimates
        noise_values = noise_assesment(iono)
        noise_array.append(noise_values)
            

                
    x,y,z = zip(*noise_array)
    c_labels = ['r' if l == 1 else 'b' for l in labels]           

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, color=c_labels)









