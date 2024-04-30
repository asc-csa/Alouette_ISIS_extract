# -*- coding: utf-8 -*-
"""
Code snippet to test out various curve fittings on the ionogram
including poylnomial, exponential,logarithmic, power, rational and logistics 


"""
#Library imports
import math
import random
import glob
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy.polynomial.polynomial as poly
#from scipy.interpolate import CubicSpline
from scipy import optimize

from extract_all_coordinates_ionogram_trace import extract_coord,map_coordinates_positions_to_values
sys.path.append('../')
from image_segmentation.segment_images_in_subdir import segment_images
from ionogram_grid_determination.grid_mapping import all_stack,get_grid_mappings

#Plots
def plot_curve_fitting(iono,adjusted_arr_coord,log_x=False, log_y=False):
    """Fits curves to the ionogram data
    
    :param path: full path of ionogram
    :type path: string
    :param adjusted_arr_coord: (Hz,km) coordinates of ionogram data
    :type adjusted_arr_coord: class: `numpy.ndarray
    :param log_x: whether a logarithmic transform should be applied to the x axis, defaults to False
    :type log_x: bool, optional
    :param log_y: whether a logarithmic transform should be applied to the y axis, defaults to False
    :type log_y: bool, optional


    """

    
    fig,(ax1,ax2) = plt.subplots(1, 2, figsize=(iono.shape[0]/64, (1.0 * iono.shape[0]/64) / 2),subplot_kw={'aspect': 'auto'})
    ax1.imshow(iono,cmap='gray',aspect="auto")
    ax1.set_title("Original " )
    ax1.axis('off')


    title_digitized = 'Digitized '
    if log_x and log_y:
        title_digitized += 'log both'
        tmp_arr = adjusted_arr_coord[:]
        adjusted_arr_coord = [(math.log(x),math.log(y)) for x, y in tmp_arr if x !=0 and y !=0]
    elif log_x:
        title_digitized += 'log x'
        tmp_arr = adjusted_arr_coord[:]
        adjusted_arr_coord = [(math.log(x),y) for x, y in tmp_arr if x != 0]
    elif log_y:
        title_digitized += 'log y'
        tmp_arr = adjusted_arr_coord[:]
        adjusted_arr_coord = [(x,math.log(y)) for x, y in tmp_arr if y !=0]

    if len(adjusted_arr_coord) != 0:
        ax2.scatter(list(zip(*adjusted_arr_coord))[0],list(zip(*adjusted_arr_coord))[1],s=1,alpha=0.05, label='raw')

    df_adjusted_arr_coord = pd.DataFrame(adjusted_arr_coord, columns=['Hz','Range'] )
    df_adjusted_arr_coord = df_adjusted_arr_coord.groupby(['Range']).median()
    df_adjusted_arr_coord.reset_index(level=0, inplace=True)
    median_adjusted_arr_coord = list(zip(df_adjusted_arr_coord['Hz'], df_adjusted_arr_coord['Range']))

    if median_adjusted_arr_coord:
        ax2.scatter(list(zip(*median_adjusted_arr_coord))[0],list(zip(*median_adjusted_arr_coord))[1],s=5, c = 'y',label='median raw')


    ax2.set_title(title_digitized)
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Altitude (km) ")
    ax2.invert_yaxis()

    min_residual = np.inf
    coeff = []
    # Try to fit a polynomial model up to degree 9
    for i in range(1,10):
        trend,[residuals, rank, singular_values, rcond]  = poly.polyfit(list(zip(*median_adjusted_arr_coord))[0],list(zip(*median_adjusted_arr_coord))[1],i, full=True)

        if len(residuals) != 0 and residuals[0] <  min_residual:
            coeff = trend

    ffit = poly.Polynomial(coeff )  

    min_x = max(min(list(zip(*median_adjusted_arr_coord))[0]),1.5)
    max_x = min(max(list(zip(*median_adjusted_arr_coord))[0]),11.5)
#    

    ax2.plot(
        np.linspace(min_x, max_x, 100),
        ffit(np.linspace(min_x, max_x, 100)),
        label=str(f'polynomial {str(i)}'),
    )
##    
#    adjusted_arr_coord_spline = sorted(adjusted_arr_coord, key=lambda coord: coord[0])
#    df_adjusted_arr_coord_spline = pd.DataFrame(adjusted_arr_coord_spline,columns = ['Hz','Range'] )
#    df_adjusted_arr_coord_spline = df_adjusted_arr_coord_spline.groupby(['Hz']).median()
#    df_adjusted_arr_coord_spline.reset_index(level=0, inplace=True)
#    adjusted_arr_coord_spline = list(zip(df_adjusted_arr_coord_spline['Hz'],df_adjusted_arr_coord_spline['Range']))
#    cs = CubicSpline(list(zip(*adjusted_arr_coord_spline ))[0],list(zip(*adjusted_arr_coord_spline ))[1])
#    ax2.plot(np.linspace(min_x,max_x,100),cs(np.linspace(min_x,max_x,100)), label='cubic spline')

    def rational(x, p, q):
        """The general rational function description (from https://stackoverflow.com/questions/29815094/rational-function-curve-fitting-in-python)
        
        p is a list with the polynomial coefficients in the numerator
        q is a list with the polynomial coefficients (except the first one)
        in the denominator
        The zeroth order coefficient of the denominator polynomial is fixed at 1.
        Numpy stores coefficients in [x**2 + x + 1] order, so the fixed
        zeroth order denominator coefficent must comes last. (Edited.)
        
        """
        return np.polyval(p, x) / np.polyval(q + [1.0], x)


    dict_functions = {'exp': lambda t,a,b: a*np.exp(b*t), 
                      'log': lambda t,a,b: a+b*np.log(t),
                      'power': lambda t,a,b,c: a*t**b + c,
                      'rational': lambda t,a,b,c,d,e: rational(t, [a, b, c], [d, e]),
                      'logistic': lambda t,a,b,c,d: a/(1+ np.exp(b + c*t))+d}

    #dictionary of functions
    # pick function minimizing error
    for fun in dict_functions:
        try:
            popt,pcov= optimize.curve_fit(dict_functions[fun],list(zip(*median_adjusted_arr_coord))[0],list(zip(*median_adjusted_arr_coord))[1], maxfev = 5000 )
        except Exception:
            continue
        ax2.plot(np.linspace(min_x,max_x,100),dict_functions[fun](np.linspace(min_x,max_x,100),*popt),label = fun)
#        
    plt.legend(loc='best')
    
if __name__ == '__main__':
    regex_subdir='E:/master/R*/[0-9]*/'
    regex_images='*.png'
    # All the subdirectory i.e. R014207948/1743-9/
    list_all_subdir = glob.glob(regex_subdir)
    
    # Randomly pick a subdirectory
    sample_subdir = list_all_subdir[random.randint(0,len(list_all_subdir) - 1)]
    
    # Segment images in the subdirectory
    df_img,_,_ =segment_images(sample_subdir, regex_images)

    # Get stack
    stack = all_stack(df_img)
    col_peaks,row_peaks,mapping_Hz, mapping_km = get_grid_mappings(stack)
    
    # Get random ionogram 
    df_sample = df_img.sample(n=1) 
    ionogram = df_sample['ionogram'].iloc[0]
    
    # Get coordinates of trace
    raw_coord, window_coord = extract_coord(ionogram ,col_peaks,row_peaks)
    
    # Get (Hz, km) coordinates from ionogram
    arr_adjusted_coord = map_coordinates_positions_to_values(window_coord,col_peaks,row_peaks,mapping_Hz,mapping_km)
    
    
    # Plot curve fittings
    plot_curve_fitting(ionogram,arr_adjusted_coord)
    plot_curve_fitting(ionogram,arr_adjusted_coord,log_x=True)
    plot_curve_fitting(ionogram,arr_adjusted_coord,log_y=True)
    plot_curve_fitting(ionogram,arr_adjusted_coord,log_x=True,log_y=True)






