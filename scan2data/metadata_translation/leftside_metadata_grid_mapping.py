# -*- coding: utf-8 -*-
"""
Determine the grid used to translate the dot and num metadata in a subdirectory
"""
# Library imports
import cv2
import numpy as np
from scipy.signal import find_peaks


# List of directories containing leftside metadata with dots
LIST_DIRECTORY_DOTS = ['R014207907F','R014207908F','R014207909F','R014207929F','R014207930F','R014207940F','R014207978F','R014207979F']

# Labelling of coordinates
LABELS_NUM = [1,2,4,8]
LABELS_CAT_DOT = ['day_1','day_2','day_3','hour_1','hour_2','minute_1','minute_2','second_1', 'second_2','station_code']
LABELS_CAT_NUM = ['satellite_number','year','day_1','day_2','day_3','hour_1','hour_2','minute_1','minute_2',
                    'second_1', 'second_2', 'station_number_1','station_number_2']
LABELS_DICT = ['dict_cat_dot','dict_num_dot','dict_cat_digit','dict_num_digit']

#Defaults for dictionary mappings of coordinates to labels
DEFAULT_DICT_CAT_DIGIT = (53,21,661) #mean_dist_default,first_peak_default,last_peak_default
DEFAULT_DICT_NUM_DIGIT = (47,41,20) #mean_dist_default,first_peak_default,dist_btw_peaks for peak detection

DEFAULT_DICT_CAT_DIGIT_F = (43,23,540) #mean_dist_default,first_peak_default,last_peak_default for those in LIST_DIRECTORY_DOTS 
DEFAULT_DICT_NUM_DIGIT_F = (40,37,20) #mean_dist_default,first_peak_default,dist_btw_peaks for peak detection for those in LIST_DIRECTORY_DOTS 

DEFAULT_DICT_CAT_DOT = (59,20,549)##mean_dist_default,first_peak_default,last_peak_default
DEFAULT_DICT_NUM_DOT = (15,32,10) #mean_dist_default,first_peak_default,dist_btw_peaks for peak detection

def extract_centroids_and_determine_type(dilated_meta,file_name,
                      min_num_pixels=50, max_number_pixels=1000,max_area_dot=120):
    
    '''Extract the coordinates of the centroid of each metadata dot/number using the connected component algorithm as well as determines if the metadata is of type dot or number
    
    :param dilated_meta: trimmed metadata (output of image_segmentation.leftside_metadata_trimming) after a rotation and dilation morphological transformation (see translate_leftside_metadata.extract_leftside_metadata )
    :type dilated_meta: class: `numpy.ndarray`
    :param file_name: full path of starting raw image ex:G:/R014207929F/431/Image0399.png
    :type file_name: str
    :param min_num_pixels: minimum number of pixels to be considered metadata dot/num, defaults to 50
    :type min_num_pixels: int, optional
    :param max_number_pixels: maximum number of pixels to be considered metadata dot/num, defaults to 1000
    :type max_number_pixels: int, optional
    :param max_area_dot: maximum median area of a single metadata components to be considered a dot, defaults to 120
    :type max_area_dot: int, optional
    :returns: col_centroids,row_centroids,is_dot : list of col ('x') positions where metadata is detected,list of row ('y') positions where metadata is detected,whether the metadata is dots
    :rtype: class: `list`,class: `list`, bool
    :raises Exception: returns np.nan,np.nan,np.nan if there is an error
    
    '''

    try:
  
        # Use connected component algorithm to determine the centroids
        _, _, stats, centroids	=	cv2.connectedComponentsWithStats(dilated_meta)
        area_centroids = stats[:,-1]
        
        # Remove centroids who are probably not associated with metadata num/dot
        centroids_metadata = centroids[np.logical_and(area_centroids > min_num_pixels, area_centroids < max_number_pixels),:]    
        col_centroids, row_centroids = zip(*centroids_metadata)
        
        # Rounding to nearest integer
        col_centroids = list(map(round,col_centroids))
        row_centroids = list(map(round,row_centroids))
        
        #Determine if dot leftside metadata 
        area_centroids = area_centroids[np.logical_and(area_centroids > min_num_pixels, area_centroids < max_number_pixels)]    
        median_area = np.median(area_centroids)
        is_dot = False
        #The line below is commented to prevent giving the dot items manually
        #if any([dir_dot in file_name for dir_dot in LIST_DIRECTORY_DOTS]) and median_area < max_area_dot:
        if median_area < max_area_dot:
            is_dot = True
        return col_centroids,row_centroids,is_dot
    except:
        return np.nan,np.nan,np.nan


def indices_highest_peaks_hist_binning(list_coord,
                   nbins=500,peak_prominence_threshold=0.2,distance_between_peaks=30):
    
    """Determines and returns the indices of the most common values in a list of coordinates using binning
    
    :param list_coord: list of positions where metadata is detected
    :type list_coord: class: `list`
    :param nbins: number of bins used for binning operation, defaults to 500
    :type nbins: int, optional
    :param peak_prominence_threshold: the threshold to detect peaks that correspond to the peaks corresponding to the most common values, defaults to 0.2
    :type peak_prominence_threshold: int, optional
    :param distance_between_peaks: the minimum number of samples between subsequent peaks corresponding to the most common values, defaults to 30
    :type distance_between_peaks: int, optional
    :returns: select_peaks,bin_edges,counts i.e. array of the indices of  peaks corresponding to the most common values, array for the bin edges after calling np.histogram, array for counts of the number of elements in each bin after calling np.histogram  
    :rtype: class: `numpy.ndarray`,class: `numpy.ndarray`,class: `numpy.ndarray`
    
    
    """
    # Transform to numpy.array
    arr_coord = np.array(list_coord )
    
    # Remove outliers
    mean_arr = np.mean(arr_coord)
    std_arr = np.std(arr_coord)
    arr_coord_no_outlier= arr_coord[np.abs(arr_coord - mean_arr) < 3 * std_arr]
    
    # Binning
    counts,bin_edges = np.histogram(arr_coord_no_outlier,bins=nbins)
    
    # Detect all peaks
    counts_norm = (counts - np.min(counts))/(np.max(counts)- np.min(counts)) #normalization
    select_peaks_idx,_ = find_peaks(counts_norm,distance = distance_between_peaks,prominence = peak_prominence_threshold)
    
    
    return select_peaks_idx,bin_edges,counts

# Check y_peaks >0
# TODO: DEFAULTS
def get_leftside_metadata_grid_mapping(list_x_dot, list_y_dot, list_x_digit, list_y_digit, dir_name,
                      difference_ratio=0.75,use_defaults=True):
    
    """Determines and returns the the mapping between coordinate values on a metadata image and metadata labels in a subdirectory, for metadata of types dot and digits, as well as returns the histogram used to generate each mapping
    
    :param list_x_dot: list of col ('x') positions where metadata of type dot is detected
    :type list_x_dot: class: `list`
    :param list_y_dot: list of row ('y') positions where metadata of type dot is detected
    :type list_y_dot: class: `list`
    :param list_x_digit: list of col ('x') positions where metadata of type digit is detected
    :type list_x_digit: class: `list`
    :param list_y_digit: list of row ('y') positions where metadata of type digit is detected
    :type list_y_digit: class: `list`
    :param dir_name: name of directory
    :type dir_name: string
    :param difference_ratio: ratio defining when to use default values, defaults to 0.5
    :type difference_ratio: int, optional
    :param use_defaults: whether to use default values, defaults to True
    :type use_defaults: bool, optional
    :returns: all_dict_mapping,all_dict_hist: dictionary of dictionaries where each dictionary correspond to a mapping between coordinates on the image and metadata labels, dictionary of histograms used to generated each dictionary in all_dict_mapping
    :rtype: dict, dict
    """
    
    # Dictionary of dictionaries that map labels to coordinate point in metadata
    all_labels = [LABELS_CAT_DOT, LABELS_NUM, LABELS_CAT_NUM, LABELS_NUM]
    all_dict_mapping = {}
    all_dict_hist = {}
    # Different protocols depending on the type of dictionary mappings
    for i, list_coord in enumerate([list_x_dot,list_y_dot,list_x_digit,list_y_digit]):
        type_dict = LABELS_DICT[i]
        labels = all_labels[i]
        try:
            if 'cat' in type_dict:
                if type_dict == 'dict_cat_digit':
                    if any([dir_dot in dir_name for dir_dot in LIST_DIRECTORY_DOTS]):
                        mean_dist_default,first_peak_default,last_peak_default=DEFAULT_DICT_CAT_DIGIT_F
                    else:
                        mean_dist_default,first_peak_default,last_peak_default=DEFAULT_DICT_CAT_DIGIT
            
                elif type_dict == 'dict_cat_dot':
                    mean_dist_default,first_peak_default,last_peak_default=DEFAULT_DICT_CAT_DOT
                try:
                    idx_peaks,bin_edges,counts = indices_highest_peaks_hist_binning(list_coord)
                    peaks = bin_edges[np.array(idx_peaks)] #coordinate values on a metadata image probably corresponding to metadata
                    
                    n_labels = len(labels)
                    first_peak = peaks[0]
                    last_peak = peaks[-1]

                    if use_defaults and abs(last_peak -last_peak_default)  > difference_ratio*mean_dist_default:
                        last_peak = last_peak_default
                    if use_defaults and abs(first_peak -first_peak_default)  > difference_ratio*mean_dist_default:
                        first_peak = first_peak_default
                        
                    mean_dist_btw_peaks = (last_peak - first_peak)/(n_labels -1)
                    list_peaks = [int(round(first_peak + i* mean_dist_btw_peaks)) for i in range(0,n_labels )]
                    
                    all_dict_mapping[type_dict] =dict(zip(list_peaks,labels))
                    all_dict_hist[type_dict] = (idx_peaks,bin_edges,counts)
                

                except:
                    last_peak = last_peak_default
                    first_peak = first_peak_default
                    mean_dist_btw_peaks = mean_dist_default
                    list_peaks = [int(round(first_peak + i* mean_dist_btw_peaks)) for i in range(0,n_labels )]
                    
                    all_dict_mapping[type_dict] =dict(zip(list_peaks,labels))
                    all_dict_hist[type_dict] = {}
                
            elif 'num' in type_dict:
                if  type_dict == 'dict_num_digit':
                    if any([dir_dot in dir_name for dir_dot in LIST_DIRECTORY_DOTS]):
                        mean_dist_default,peak_0_default,dist_btw_peaks = DEFAULT_DICT_NUM_DIGIT_F
                    else:
                        mean_dist_default,peak_0_default,dist_btw_peaks = DEFAULT_DICT_NUM_DIGIT
                elif type_dict == 'dict_num_dot':
                    mean_dist_default,peak_0_default,dist_btw_peaks= DEFAULT_DICT_NUM_DOT

                    
                try:
                    idx_peaks,bin_edges,counts = indices_highest_peaks_hist_binning(list_coord,peak_prominence_threshold=0.3,nbins=100,distance_between_peaks=dist_btw_peaks)
                
                    peaks = bin_edges[np.array(idx_peaks)]                
                    peak_0 = peaks[0]
                    if use_defaults and abs(peak_0 -peak_0_default)  > difference_ratio*mean_dist_default:
                        peak_0 = peak_0_default
                
                    # only first three peaks are deemed relevant
                    if len(peaks) < 3:
                        max_idx = 2
                    else:
                        max_idx = 3
                
                    mean_dist_btw_peaks = np.mean([peaks[i+1]-peaks[i] for i in range(0,max_idx)])
                    if use_defaults and abs(mean_dist_btw_peaks - mean_dist_default)  > difference_ratio*dist_btw_peaks:
                        mean_dist_btw_peaks = mean_dist_default
                    list_peaks = [int(round(peak_0 + i* mean_dist_btw_peaks)) for i in range(0,len(labels))]
                
                    all_dict_mapping[type_dict] =dict(zip(list_peaks,labels))
                    all_dict_hist[type_dict] = (idx_peaks,bin_edges,counts)
                except:
                    peak_0 = peak_0_default
                    mean_dist_btw_peaks = mean_dist_default
                    list_peaks = [int(round(peak_0 + i* mean_dist_btw_peaks)) for i in range(0,len(labels))]
                    all_dict_mapping[type_dict] =dict(zip(list_peaks,labels))
                    all_dict_hist[type_dict] =  {}
        except:
            all_dict_mapping[type_dict] ={}
            all_dict_hist[type_dict] =  {}

            
    return all_dict_mapping,all_dict_hist

