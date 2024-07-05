#ISIS Full MetaData Analysis - Jose Pasillas May 2024

#Required imports 
import pandas as pd
import numpy as np
import cv2
import os, gc, sys
from random import randrange
import time
from datetime import datetime
from optparse import OptionParser

#Required imports for KERAS
import string 
from PIL import Image
import string
import tempfile
from matplotlib import pyplot as plt

L_drive = 'L:'

## CHANGE PATH BELOW TO YOUR VIRTUAL ENVIRONMENT
sys.path.insert(0,"V:\\PATH...\Python\envs\\") 

import tensorflow as tf
import keras_ocr

print('tensorflow version (should be 2.10.* for GPU compatibility) is: ' , tf.__version__)
if len(tf.config.list_physical_devices('GPU')) != 0:
    print('GPU in use for tensorflow')
else:
    print('CPU in use for tensorflow')

#To run this script : e.g Data_Analysis_ISIS.py --username YOUR_USERNAME --isis 1 (for ISIS batch 2 run by jpasillas)
#Script defaults to ISIS Batch 2 

parser = OptionParser()

parser.add_option('-u', '--username', dest='username', 
        default='YOUR_USERNAME', type='str', 
        help='CSA Network username, default=%default.')

parser.add_option('--isis', dest='isis', 
        default='1', type='str', 
        help='ISIS batch, default=%default.')

(options, args) = parser.parse_args()

if options.isis == '2':
     #ISIS BATCH 2 CHOSEN
     directory_path = L_drive + 'DATA/ISIS/ISIS_102000056114/'
     batch_size = 801

    #Log Directory, do not change
     logDir = L_drive + '/DATA/ISIS/ISIS_Test_Data_Analysis/BATCH_2/04_log/'
     #Path to save results, do not change
     resultDir = L_drive + '/DATA/ISIS/ISIS_Test_Data_Analysis/BATCH_2/05_results/'
     my_path = logDir + 'ISIS_2_Directory_Subdirectory_List.csv'

elif options.isis == '3':
     #ISIS BATCH 3/RAW UPLOAD CHOSEN
     directory_path = L_drive + '/DATA/ISIS/raw_upload_20230421/'
     batch_size = 359

    #Log Directory, do not change
     logDir = L_drive + '/DATA/ISIS/ISIS_Test_Data_Analysis/BATCH_3/04_log/'
     #Path to save results, do not change
     resultDir = L_drive + '/DATA/ISIS/ISIS_Test_Data_Analysis/BATCH_3/05_results/'
     my_path = logDir + 'ISIS_3_Directory_Subdirectory_List.csv'

else:
     #ISIS BATCH 1 
     directory_path = L_drive + '/DATA/ISIS/ISIS_101300030772/'
     batch_size = 1720

    #Log Directory, do not change
     logDir = L_drive + '/DATA/ISIS/ISIS_Test_Data_Analysis/BATCH_1/04_log/'
     #Path to save results, do not change
     resultDir = L_drive + '/DATA/ISIS/ISIS_Test_Data_Analysis/BATCH_1/05_results/'
     my_path = logDir + 'ISIS_1_Directory_Subdirectory_List.csv'

cropped_too_soon = L_drive + "/DATA/ISIS/contractor_error_reports/CSA-AMS Comparison/CSAnotAMS_allmerged.csv"
cropped_too_soon_df = pd.read_csv(cropped_too_soon)

#station names and location 
station_log_dir = L_drive + '/DATA/ISIS/ISIS_Test_Data_Analysis/Station_Number_Name_Location.csv'
station_df = pd.read_csv(station_log_dir)


def draw_random_subdir():
    '''
    Definition: Draw a directory and subdirectory, that is not currently in progress or has already been processed and updates the status
    of that row 
      
    Arguments: None

    Returns: A directory, subdirectory and the row number at which these are found 
        
    '''
    #if os.path.exists(my_path):
    if 1 == 1 :
            try:
                full_dir_df = pd.read_csv(my_path)
                print(full_dir_df)
                ind = randrange(len(full_dir_df))
                directory = full_dir_df['Directory'][ind]
                subdir = full_dir_df['Subdirectory'][ind]

                if (full_dir_df['Status'][ind]) == "Not Processed":
                    full_dir_df.loc[ind, "Status"] = "In Progress"
                    full_dir_df.to_csv(my_path, index=False)
                    return directory, subdir, ind

                elif (full_dir_df['Status'][ind]) == "In Progress":
                    print('Current subdirectory', subdir, 'being processed already, moving on to the next one')
                    directory, subdir, ind = draw_random_subdir()
                    return directory, subdir, ind
                    
                    
                elif (full_dir_df['Status'][ind]) == "Processed":
                    print("Current subdirectory", subdir, "already processed, moving on to the next one")
                    directory, subdir, ind = draw_random_subdir()
                    return directory, subdir, ind
                    

            except (OSError, PermissionError) as e:
                print(my_path, 'currently being used, pausing for 30 seconds before another attempt')
                time.sleep(30)
               
def update_my_log_file(ind):
    '''
    Definition: Updates the status of the given index row for dir and subdir as "Processed"
      
    Arguments:
        ind: Index row for processed directory and subdirectory

    Returns: None
        
    '''
    if os.path.exists(my_path):
        try:
            full_dir_df = pd.read_csv(my_path)
            
            if (full_dir_df['Status'][ind]) == "In Progress":
                    full_dir_df.loc[ind, "Status"]= "Processed"
                    full_dir_df.to_csv(my_path, index=False)
            else:
                print("Error with this path - check if dir and subdir at row", str(ind), "has been processed")
                 
        except (OSError, PermissionError) as e:
                print(my_path, 'currently being used, pausing for 30 seconds before another attempt')
                time.sleep(30)

def get_station_info(ind):
    '''
    Definition: This function takes in a station number (read from an ionogram), cross references it with 
    a station information csv and returns the station location, ID, Latitude and Longitude 
 
    Arguments:
        ind: int corresponding to station number 

    Returns: 4 strings corresponding to the location, latitude, longitiude and ID of the given station number 

    '''    
    for i in range(len(station_df)):
        if station_df['Number'][i] == str(ind):
            station_location = station_df['Location'][i]
            station_lat =  station_df['Latitude'][i]
            station_lon = station_df['Longitude'][i]
            station_ID =  station_df['Station ID'][i]
            return station_location, station_lat, station_lon, station_ID
        else:
            station_ID = station_location = station_lon = station_lat = 0
    
    return station_location, station_lat, station_lon, station_ID

def crop_image_by_threshold(image, threshold=0.8):
    """
    Crop an image by removing the top and bottom borders based on a threshold of row sums.

    Args:
    image (np.array): The input image in grayscale or color.
    threshold (float): The threshold percentage to identify borders, default is 0.8.

    Returns:
    np.array: The cropped image.
    """

    # Convert the image to grayscale if it is in color
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Sum pixel values along each row
    row_sums = np.sum(gray, axis=1)

    # Normalize the row sums
    row_sums_normalized = row_sums / np.max(row_sums)

    # Determine rows that exceed the threshold
    valid_rows = np.where(row_sums_normalized > threshold)[0]
    
   
                             
    if len(valid_rows) <= 0:
        # If no valid rows are found, return the original image
        return image

    # Find the top and bottom borders using the first and last valid rows
    top_border_index = valid_rows[0]
    bottom_border_index = valid_rows[-1]

    # Crop the image to remove the top and bottom borders
    
    cropped_image = image[top_border_index:bottom_border_index+1, :]
    crop_size = [top_border_index, bottom_border_index]
    
    if len(valid_rows) <crop_size[0]:
        cropped_image=image

    return cropped_image, crop_size


def extract_roi_with_template_matching(img_test, img_crop, minimum_correlation, size_increase):
    """
    Extract a region of interest (ROI) from a test image based on template matching.

    Args:
    img_test (np.array): The test image in which to find the template.
    img_crop (np.array): The template image to find in the test image.
    minimum_correlation (float): The minimum correlation coefficient to accept a match.
    size_increase (float): The percentage by which to increase the size of the bounding box around the match.

    Returns:
    np.array: The extracted ROI, or None if no suitable match is found.
    """

    # print('image_test02',img_test.shape)
    # print('image_crop02',img_crop.shape)
    if img_test.shape[1] < img_crop.shape[1]:
        img_test = img_crop
    # Perform template matching
    correlation_output = cv2.matchTemplate(img_test, img_crop, cv2.TM_CCOEFF_NORMED)
    
 
    
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(correlation_output)
    # Check if the best match meets the minimum correlation threshold
    if max_val < minimum_correlation:
        print("No match meets the minimum correlation threshold.")
        return False, False, max_val

    # Calculate the original bounding box dimensions
    bbox_width = img_crop.shape[1]
    bbox_height = img_crop.shape[0]

    # Calculate the increase in dimensions
    increase_width = int(bbox_width * size_increase)
    increase_height = int(bbox_height * size_increase)

    # Adjust the bounding box to include the increased size
    start_point = (max_loc[0] - increase_width // 2, max_loc[1] - increase_height // 2)
    end_point = (start_point[0] + bbox_width + increase_width, start_point[1] + bbox_height + increase_height)

    # Ensure the adjusted bounding box remains within the image boundaries
    ext_start_point = (max(0, start_point[0]), max(0, start_point[1]))
    ext_end_point = (min(img_test.shape[1], end_point[0]), min(img_test.shape[0], end_point[1]))

    # Extract and return the ROI from the test image
    ext_roi = img_test[ext_start_point[1]:ext_end_point[1], ext_start_point[0]:ext_end_point[0]]
    
    bounding_box = [ext_start_point[0], ext_start_point[1], ext_end_point[0], ext_end_point[1]]
    
    return ext_roi, bounding_box, max_val

# this function has potential room for improvement
# thresholds & tolerance are fairly stable at a range of values
def detect_common_lines(image, bbox, crop_size, img_format, threshold=180, tolerance=2):
    # Function to detect lines using Hough Transform
    def hough_lines_detection(img):
        blur = cv2.GaussianBlur(img, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 150)
        hough_lines = []
        if lines is not None:
            for line in lines:
                rho, theta = line[0]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                hough_lines.append((x1, y1, x2, y2))
        return hough_lines

    # Function to detect lines based on average color thresholding
    def simple_line_detection(img):
        average_color = np.mean(img)
        height, width = img.shape[:2]
        detected_lines = {'horizontal': [], 'vertical': []}
        # Detecting horizontal lines
        for y in range(height):
            row = img[y, :]
            if np.mean(row <= threshold) > 0.5:
                detected_lines['horizontal'].append((0, y, width - 1, y))
        # Detecting vertical lines
        for x in range(width):
            col = img[:, x]
            if np.mean(col <= threshold) > 0.5:
                detected_lines['vertical'].append((x, 0, x, height - 1))
        return detected_lines

    # Convert lines to a unified format (if needed) and find common lines
    def match_lines(hough, simple):
        common_lines = {'horizontal': [], 'vertical': []}
        # Horizontal lines
        for line1 in hough:
            for line2 in simple['horizontal']:
                if abs(line1[1] - line2[1]) < tolerance:
                    common_lines['horizontal'].append(line2)
        # Vertical lines
        for line1 in hough:
            for line2 in simple['vertical']:
                if abs(line1[0] - line2[0]) < tolerance:
                    common_lines['vertical'].append(line2)
        return common_lines

    hough_lines = hough_lines_detection(image)
    simple_lines = simple_line_detection(image)
    common_lines = match_lines(hough_lines, simple_lines)

    common_horizontal = [(0, 106-bbox[1]-crop_size[0], image.shape[1], 106-bbox[1]-crop_size[0]), 
     (0, 179-bbox[1]-crop_size[0], image.shape[1], 179-bbox[1]-crop_size[0]), 
     (0, 250-bbox[1]-crop_size[0], image.shape[1], 250-bbox[1]-crop_size[0]), 
     (0, 321-bbox[1]-crop_size[0], image.shape[1], 321-bbox[1]-crop_size[0])]
    
    common_horizontal = [line for line in common_horizontal if line[1] <= image.shape[0]]
    
    if img_format: 
        common_lines['horizontal'] = common_horizontal

    return common_lines


def color_lines_on_image(image, common_lines):
    """
    Fills regions of line with the average color of an image
    
    Args:
    image (np.array): The input image in grayscale or color.
    common_lines (dicionary): vertical and horizontal lines to be colored

    Returns:
    image (np.array): An imgage with lines colored with average color of image
    """
    
    # Function to compute the average color of the image
    def compute_average_color(img):
        return np.mean(img, axis=(0, 1))

    # Average color calculation
    average_color = compute_average_color(image)
    
    # Prepare average color in correct format if image is not grayscale
    if len(image.shape) == 3:
        average_color = (int(average_color[0]), int(average_color[1]), int(average_color[2]))
    else:
        average_color = int(average_color)
    
    # Create a copy of the original image to modify
    output_image = image.copy()

    # Draw lines with the average color
    for line_type in common_lines:
        for line in common_lines[line_type]:
            if line_type == 'horizontal':
                cv2.line(output_image, (line[0], line[1]), (line[2], line[3]), average_color, 2)
            elif line_type == 'vertical':
                cv2.line(output_image, (line[0], line[1]), (line[2], line[3]), average_color, 2)

    return output_image

# parameters here could be adjusted to improve connectivity of discontinous trace
def detect_single_rightmost_point(image, gaussian_blur_size=5, canny_thresholds=(50, 150),
                                  dilation_kernel_size=5, dilation_iterations=2): 
    """
    Determines the rightmost point based on dilation of the largest object in image
    
    Args:
    image (np.array): The input image in grayscale or color.

    Returns:
    point (tuple): The rightmost detected point
    """    
    
    # Apply Gaussian Blur, useful for canny edge
    blurred_image = cv2.GaussianBlur(image, (gaussian_blur_size, gaussian_blur_size), 0)
    
    # Apply Canny Edge Detection
    edges = cv2.Canny(blurred_image, canny_thresholds[0], canny_thresholds[1])
    
    # Morphological Dilation
    dilation_kernel = np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8)
    dilated_image = cv2.dilate(edges, dilation_kernel, iterations=dilation_iterations)
    
    # Find contours
    contours, _ = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Assuming the largest contour is the curve of interest
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Simplify the largest contour to reduce the number of points
    epsilon = 0.01 * cv2.arcLength(largest_contour, True)
    approx_curve = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    # Find the rightmost point of the largest contour
    rightmost_point = tuple(approx_curve[approx_curve[:, :, 0].argmax()][0])
        
    # Return the location of the rightmost detected pixel
    return rightmost_point

def preprocess_lines(common_lines, proximity_threshold):
    """
    Combines any duplicate lines
    
    Args:
    common_lines (dicitionary): Serise of vertical and horizontal lines. 
    proximity_threshold (int): To distance between lines to combine

    Returns:
    common_lines (dicitionary): Serise of vertical and horizontal lines. 
    """
    
    def merge_close_lines(sorted_lines, proximity_threshold, is_horizontal=False):
        merged_lines = []
        current_line = sorted_lines[0]

        for next_line in sorted_lines[1:]:
            # Check the proximity based on x for vertical lines, y for horizontal lines
            if is_horizontal:
                distance = abs(next_line[1] - current_line[1])  # Use y-coordinate for horizontal lines
            else:
                distance = abs(next_line[0] - current_line[0])  # Use x-coordinate for vertical lines

            # If the next line is close to the current line, merge them
            if distance <= proximity_threshold:
                current_line = tuple(
                    np.mean([current_line, next_line], axis=0).astype(int).tolist()
                )
            else:
                merged_lines.append(current_line)
                current_line = next_line

        # Append the last processed line
        merged_lines.append(current_line)
        return merged_lines

    # Sort and preprocess vertical and horizontal lines separately
    vertical_lines_sorted = sorted(common_lines['vertical'], key=lambda x: x[0])
    horizontal_lines_sorted = sorted(common_lines['horizontal'], key=lambda x: x[1])

    common_lines['vertical'] = merge_close_lines(vertical_lines_sorted, proximity_threshold)
    common_lines['horizontal'] = merge_close_lines(horizontal_lines_sorted, proximity_threshold, is_horizontal=True)

    return common_lines

# the calculation method for ratio could be adjusted to account if there is an unexpected amount of detected lines
def analyze_point_against_lines(point, common_lines, proximity_threshold):
    """
    Determines statistics agaisnt detected lines
    
    Args:
    point (tupple): The input image in grayscale or color.
    common_lines (dicitionary): Serise of vertical and horizontal lines. 
    proximity_threshold (int): To distance between lines to combine

    Returns:
    df (dataframe): Dataframe of results
    """
    
    try:
        # Preprocess the lines
        common_lines = preprocess_lines(common_lines, proximity_threshold)
        x_point, y_point = point

        # Sort and filter lines
        vertical_lines_sorted = sorted(common_lines['vertical'], key=lambda x: x[0])
        horizontal_lines_sorted = sorted(common_lines['horizontal'], key=lambda x: x[1])
        vertical_lines_to_left = [line for line in vertical_lines_sorted if line[0] < x_point]
        horizontal_lines_above = [line for line in horizontal_lines_sorted if line[1] < y_point]

        # Calculate number of lines
        num_vertical_lines_left = len(vertical_lines_to_left)
        num_horizontal_lines_above = len(horizontal_lines_above)

        # Calculate vertical spacing ratio
        vertical_spacing_ratio = np.nan  # Default to NaN
        if num_vertical_lines_left >= 2:
            vertical_spacing = vertical_lines_to_left[-1][0] - vertical_lines_to_left[-2][0]
            if vertical_spacing > 0:
                vertical_spacing_ratio = (x_point - vertical_lines_to_left[-1][0]) / vertical_spacing
            # Handle unexpected negative ratio
            if vertical_spacing_ratio < 0 or vertical_spacing_ratio > 1:
                vertical_spacing_ratio = np.nan

        # Calculate horizontal spacing ratio
        horizontal_spacing_ratio = np.nan  # Default to NaN
        if num_horizontal_lines_above >= 2:
            horizontal_spacing = horizontal_lines_above[-1][1] - horizontal_lines_above[-2][1]
            if horizontal_spacing > 0:
                horizontal_spacing_ratio = (y_point - horizontal_lines_above[-1][1]) / horizontal_spacing
            # Handle unexpected negative ratio
            if horizontal_spacing_ratio < 0 or horizontal_spacing_ratio > 1:
                horizontal_spacing_ratio = np.nan

    except Exception as e:
        # In case of any other unexpected error, log it and set all values to NaN
        print(f"An error occurred: {e}")
        num_vertical_lines_left = np.nan
        num_horizontal_lines_above = np.nan
        vertical_spacing_ratio = np.nan
        horizontal_spacing_ratio = np.nan

    # Create a pandas DataFrame to return results
    df = pd.DataFrame({
        'detected_point': [point],
        'vertical_count': [num_vertical_lines_left],
        'horizontal_count': [num_horizontal_lines_above],
        'vertical_ratio': [vertical_spacing_ratio],
        'horizontal_ratio': [horizontal_spacing_ratio]
    })

    return df

# helper function for visual accuarcy assesment
def visualize_and_save(image, common_lines, point, image_path, detection=True, output_folder='results_testing_jp'):
    """
    Visualize an image with lines and a point plotted on it, or create a blank image based on the detection flag,
    and save the output to a specified directory.

    Args:
    image (np.array): The image on which to plot lines and points if detection is True.
    common_lines (dict): Dictionary containing 'horizontal' and 'vertical' lines.
    point (tuple): The coordinates of the point to plot.
    image_path (str): Original path of the image.
    detection (bool): If True, plot with lines and point, otherwise create a blank black image.
    output_folder (str): Directory to save the modified image.
    """

    # Ensure output directory exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Create the output image path
    image_filename = os.path.basename(image_path)
    output_image_path = os.path.join(output_folder, image_filename)

    if detection:
        # Start plotting if detection is True
        plt.figure(figsize=(5, 3))
        plt.imshow(image, cmap='gray')
        # Plot horizontal and vertical lines
        for line in common_lines['horizontal']:
            plt.plot([line[0], line[2]], [line[1], line[3]], 'r')
        for line in common_lines['vertical']:
            plt.plot([line[0], line[2]], [line[1], line[3]], 'b')
        # Plot the point
        plt.scatter(*point, color='yellow', s=40, edgecolors='black')  # Use scatter to mark the point
        plt.title(image_path)
        plt.axis('off')  # Optionally turn off the axis.
        # Save the figure
        plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0.1)
        plt.close()  # Close the figure to free up memory
    else:
        # Create a 256x256 black image if detection is False
        black_image = np.zeros((256, 256), dtype=np.uint8)
        cv2.imwrite(output_image_path, black_image)

    print(f"Image saved to {output_image_path}")
    
def process_images(full_paths, minimum_correlation=0.5, size_increase=0.25, threshold=0.8, line_width=3):
    """
    Process a list of image paths to extract, analyze, and visualize ROIs based on template matching.

    Args:
    full_paths (list): List of paths to the images to process.
    img_crop_path (str): Path to the template image.
    minimum_correlation (float): Minimum correlation to consider a match valid.
    size_increase (float): Factor to increase the size of the ROI around the matched template.
    threshold (float): Threshold for initial cropping of the image.

    Returns:
    pd.DataFrame: DataFrame containing the results of the processing.
    """

    print(os.getcwd())
    
    # The paths may need to be updated depending where the templates are being stored.
    # This assumes that the templates are in the current working directory. 
    template_a_1 = cv2.imread('template_a_1.png', cv2.IMREAD_GRAYSCALE)
    template_a_2 = cv2.imread('template_a_2.png', cv2.IMREAD_GRAYSCALE)
    template_a_3 = cv2.imread('template_a_3.png', cv2.IMREAD_GRAYSCALE)
    template_a_4 = cv2.imread('template_a_4.png', cv2.IMREAD_GRAYSCALE)
    
    template_b_1 = cv2.imread('template_b_1.png', cv2.IMREAD_GRAYSCALE)
    template_b_2 = cv2.imread('template_b_2.png', cv2.IMREAD_GRAYSCALE)
    template_b_3 = cv2.imread('template_b_3.png', cv2.IMREAD_GRAYSCALE)
    template_b_4 = cv2.imread('template_b_4.png', cv2.IMREAD_GRAYSCALE)
    
    template_c_1 = cv2.imread('template_c_1.png', cv2.IMREAD_GRAYSCALE)
    template_c_2 = cv2.imread('template_c_2.png', cv2.IMREAD_GRAYSCALE)
    template_c_3 = cv2.imread('template_c_3.png', cv2.IMREAD_GRAYSCALE)
    template_c_4 = cv2.imread('template_c_4.png', cv2.IMREAD_GRAYSCALE)
    
    template_ghost = cv2.imread('template_ghost.png', cv2.IMREAD_GRAYSCALE)
    
    if template_a_1 or template_a_2 or template_a_3 or template_a_4 or \
    template_b_1 or template_b_2 or template_b_3 or template_b_4 or \
    template_c_1 or template_c_2 or template_c_3 or template_c_4 is None:
        raise FileNotFoundError("Template image not found at the specified path.")

    results = []  # List to store all DataFrame results

    for image_path in full_paths:
         
        img_test = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img_test is None:
            print(f"Failed to load image: {image_path}")
            continue
            
        if img_test.shape[0] == 410:
            img_format = True 
        else:
            img_format = False 
                
        cropped_image, crop_size = crop_image_by_threshold(img_test, threshold=threshold)
        
        if cropped_image.shape[0] <= template_a_1.shape[0] or cropped_image.shape[0] <= template_a_2.shape[0] or \
            cropped_image.shape[0] <= template_a_3.shape[0] or cropped_image.shape[0] <= template_a_4.shape[0] or \
            cropped_image.shape[0] <= template_b_1.shape[0] or cropped_image.shape[0] <= template_b_2.shape[0] or \
            cropped_image.shape[0] <= template_b_3.shape[0] or cropped_image.shape[0] <= template_b_4.shape[0] or \
            cropped_image.shape[0] <= template_c_1.shape[0] or cropped_image.shape[0] <= template_c_2.shape[0] or \
            cropped_image.shape[0] <= template_c_3.shape[0] or cropped_image.shape[0] <= template_c_4.shape[0]:
            cropped_image = img_test
            
        # Determine the average pixel DN of the film to broadly classify to type of Black: 1, Gray: 2, White: 3    
        average_intensity = np.mean(img_test)
        
        # Return correlations for the 4 types of ionograms. 
        if average_intensity <= 100:
            film_type = 1
            _, _, corr_1 = extract_roi_with_template_matching(cropped_image, template_c_1, minimum_correlation, size_increase)
            _, _, corr_2 = extract_roi_with_template_matching(cropped_image, template_c_2, minimum_correlation, size_increase)
            _, _, corr_3 = extract_roi_with_template_matching(cropped_image, template_c_3, minimum_correlation, size_increase)
            _, _, corr_4 = extract_roi_with_template_matching(cropped_image, template_c_4, minimum_correlation, size_increase)
            _, _, corr_5 = extract_roi_with_template_matching(cropped_image, template_ghost, minimum_correlation, size_increase)
            
        elif 100 < average_intensity <= 180:
            film_type = 2
            _, _, corr_1 = extract_roi_with_template_matching(cropped_image, template_b_1, minimum_correlation, size_increase)
            _, _, corr_2 = extract_roi_with_template_matching(cropped_image, template_b_2, minimum_correlation, size_increase)
            _, _, corr_3 = extract_roi_with_template_matching(cropped_image, template_b_3, minimum_correlation, size_increase)
            _, _, corr_4 = extract_roi_with_template_matching(cropped_image, template_b_4, minimum_correlation, size_increase)
            _, _, corr_5 = extract_roi_with_template_matching(cropped_image, template_ghost, minimum_correlation, size_increase)
            
        elif average_intensity > 180:    
            film_type = 3
            roi, bbox, corr_1 = extract_roi_with_template_matching(cropped_image, template_a_1, minimum_correlation, size_increase)
            _, _, corr_2 = extract_roi_with_template_matching(cropped_image, template_a_2, minimum_correlation, size_increase)
            _, _, corr_3 = extract_roi_with_template_matching(cropped_image, template_a_3, minimum_correlation, size_increase)
            _, _, corr_4 = extract_roi_with_template_matching(cropped_image, template_a_4, minimum_correlation, size_increase)
            _, _, corr_5 = extract_roi_with_template_matching(cropped_image, template_ghost, minimum_correlation, size_increase)
    

        if roi is not False:
            common_lines = detect_common_lines(roi, bbox, crop_size, img_format)
            image_colored = color_lines_on_image(roi, common_lines)
            single_rightmost_point = detect_single_rightmost_point(image_colored)  # Ensure this function is defined
            df_result = analyze_point_against_lines(single_rightmost_point, common_lines, line_width)
            df_result['image_path'] = image_path
            df_result['film_type'] = film_type
            df_result['corr_1'] = corr_1
            df_result['corr_2'] = corr_2
            df_result['corr_3'] = corr_3
            df_result['corr_4'] = corr_4
            df_result['corr_5'] = corr_5
            df_result['file_name'] = os.path.basename(image_path)

            if 'detected_point' in df_result:
                point = df_result['detected_point'][0]
                adjusted_point = (point[0] + bbox[0], point[1] + bbox[1] + crop_size[0])
                df_result['detected_point'] = [adjusted_point]
                #visualize_and_save(image_colored, common_lines, single_rightmost_point, image_path, detection=True)
            
            results.append(df_result)
        else:
            # If ROI is None, append a DataFrame with NaN values and only the image path
            #visualize_and_save(cropped_image, {}, (0, 0), image_path, detection=False)
            df_result = pd.DataFrame({
                'detected_point': [np.nan],
                'vertical_count': [np.nan],
                'horizontal_count': [np.nan],
                'vertical_ratio': [np.nan],
                'horizontal_ratio': [np.nan],
                'image_path': [image_path],
                'film_type': film_type,
                'corr_1': corr_1,
                'corr_2': corr_2,
                'corr_3': corr_3,
                'corr_4': corr_4,
                'corr_5': corr_5,
                'file_name':[os.path.basename(image_path)]
            })
            results.append(df_result)

    # Concatenate all results into a single DataFrame
    final_df = pd.concat(results, ignore_index=True)
    return final_df

#Start processing 
stop_condition = False

while stop_condition == False:
    start = time.time()
    
    #Get number of processed subdirs 
    if os.path.exists(logDir + 'Process_Log.csv'):
        try:
            my_log_file = pd.read_csv(logDir + 'Process_Log.csv')
            subdirs_processed = len(my_log_file['Subdirectory'].drop_duplicates())
            dirs_processed = len(my_log_file['Directory'].drop_duplicates())

        except (OSError, PermissionError) as e:
            print(logDir + 'Process_Log_OCR.csv', 'currently being used, pausing for 30 seconds before another attempt')
            time.sleep(30)

        #get remaining subdirs        
        subdir_rem = batch_size - subdirs_processed

        #Check stop conditions
        if subdir_rem < 2:
            print('Stop!')
            stop_condition = True
   
    #Get directory and subdirectory path to process and current row index
    directory, subdirectory, curr_row_index = draw_random_subdir()
    subdir_path_end = directory + '/' + subdirectory + '/'

    print('')
    print('Processing ' + subdir_path_end + ' subdirectory')
    print(str(subdir_rem) + ' subdirectories to go!')

    #Cross reference images that were flagged as cropped too soon & remove them
    flagged_dir = cropped_too_soon_df.loc[cropped_too_soon_df['Directory'] == directory]
    flagged_subdir = flagged_dir.loc[flagged_dir['Subdirectory'] == subdirectory]

    #Get all images from chosen directory and subdirectory path
    img_fns = []
    for file in os.listdir(directory_path + subdir_path_end):
        if file.endswith('.png'):
            if (flagged_subdir['filename'] == file).any():
                continue #Cross reference images that were flagged as cropped too soon & remove them
            img_fns.append(directory_path + subdir_path_end + file)
            num_images = len(img_fns)

    df_read = pd.DataFrame()
    df_notread = pd.DataFrame()

           
    df_read = process_images(img_fns, minimum_correlation=0.5)
        
    #Saving results:
    my_temp_path = resultDir + directory
    if not os.path.exists(my_temp_path):
        path = os.path.join(resultDir, directory)
        os.makedirs(path)
    
    df_read['Directory'] = directory
    df_read['Subdirectory'] = subdirectory
    df_read.to_csv(resultDir + directory + '/' +  'data_analysis_' + subdirectory + '.csv', index=False)
    if len(df_notread) > 0:
        df_notread.to_csv(resultDir + directory + '/' +  'LOSS_data_analysis_' + subdirectory + '.csv', index=False)

    print('Dir:', directory, 'Subdir:', subdirectory, "results saved to csv!")

    #update status for current path of dir and subdir
    update_my_log_file(curr_row_index)
    print("Status updated!")

    #Processing time for one subdirectory
    end = time.time()
    t = end - start
    print('Processing time for subdirectory: ' + str(round(t/60, 1)) + ' min')
    print('Processing rate: ' + str(round(t/len(img_fns), 2)) + ' s/img')
    print('')

    #Record performance
    df_result_ = pd.DataFrame({
        'Directory': directory,
        'Subdirectory': subdirectory,
        '# images' : num_images,
        'Process_time': t,
        'Process_timestamp': datetime.fromtimestamp(end),
        #'User': options.username
        'User': 'jose'
    }, index=[0])

    if os.path.exists(logDir + 'Process_Log.csv'):
        df_log = pd.read_csv(logDir + 'Process_Log.csv')
        df_update = pd.concat([df_log, df_result_], axis=0, ignore_index=True)
        df_update.to_csv(logDir + 'Process_Log.csv', index=False)

    else:
        if len(df_result_) > 0:
            df_result_.to_csv(logDir + 'Process_Log_OCR.csv', index=False)
            
    #Backup 'process_log' (10% of the time), garbage collection
    if randrange(10) == 7:
        df_log = pd.read_csv(logDir + 'Process_Log.csv')
        datetime_str = datetime.now().strftime("%Y%m%d_%Hh%M")
        os.makedirs(logDir + 'backups/', exist_ok=True)
        df_log.to_csv(logDir + 'backups/' + 'process_log_OCR-' + datetime_str + '.csv', index=False)
        gc.collect()
