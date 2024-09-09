# -*- coding: utf-8 -*-
"""
Data Labelling of Ionograms
@author: Jackson Cooper
"""
import os
import cv2
import pandas as pd
from tkinter import Tk, Toplevel, Label, Entry, Button, StringVar, Scale, HORIZONTAL, Radiobutton, IntVar, Frame

def load_images_from_folder(folder):
    """ Load all image filenames from the specified directory. """
    return [os.path.join(folder, f) for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]


def load_images_from_path(csv_path, N, result_path):
    """ Load N random unique image filenames from a specified CSV file """
    df = pd.read_csv(csv_path)

    if 'path' not in df.columns:
        return []
    
    if N > df.shape[0]:
        N = df.shape[0]
    
    sampled_df = df['path'].sample(n=N, replace=False).tolist()

    # Checks if the images to be sampled have already been observed
    if os.path.exists(result_path):
        try:
            result_df = pd.read_csv(result_path)
            if 'path' in result_df.columns:
                existing_paths = set(result_df['path'].tolist())
                sampled_df = [path for path in sampled_df if path not in existing_paths]
            del result_df
        except:
            print('Results is empty.')

    return sampled_df


def load_images_from_name(csv_path, N, result_path):
    """ Load N random unique image filenames from a specified Excel file with a column 'File name', excluding paths already in result_path. """
    df = pd.read_csv(csv_path)

    root_dir_1 = 'L:/DATA/ISIS/ISIS_101300030772/'
    root_dir_2 = 'L:/DATA/ISIS/ISIS_102000056114/'
    
    if 'File name' not in df.columns:
        return []

    if N > df.shape[0]:
        N = df.shape[0]
    
    sampled_df = df['File name'].sample(n=N, replace=False).tolist()
    
    # Create paths based on the starting character
    sampled_df = [
        os.path.join(root_dir_1, file_name) if file_name.startswith('b') else os.path.join(root_dir_2, file_name)
        for file_name in sampled_df
    ]

    if os.path.exists(result_path):
        try:
            result_df = pd.read_csv(result_path, na_values=[], keep_default_na=False)
            if 'path' in result_df.columns:
                existing_paths = set(result_df['path'].tolist())
                sampled_df = [path for path in sampled_df if path not in existing_paths]
            del result_df
        except:
            print('Results file is empty or invalid.')
            
    return sampled_df


class LabelDialog(Toplevel):
    """ Custom dialog for labeling images with additional point selection functionality """
    def __init__(self, parent, title, prompt, img_path):
        
        # Initialization of variables to be used in GUI
        super().__init__(parent)
        self.parent = parent
        self.img_path = img_path
        self.original_img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        self.processed_img = self.original_img.copy()
        self.points = {'grid_point': None, 'bottom_point': None, 'right_point': None}
        self.point_mode = None
        self.user_closed = False

        self.title(title)
        self.protocol("WM_DELETE_WINDOW", self.on_exit)  
        self.create_widgets(prompt)
        self.display_image()

    def adjust_contrast(self, value):
        """ Adjust the image contrast based on the slider value """
        factor = float(value)
        self.processed_img = cv2.convertScaleAbs(self.original_img, alpha=factor, beta=0)
        self.redraw_image()

    def apply_image_processing(self):
        """ Apply selected image processing based on the radio button selection and update the image display. """
        choice = self.process_var.get()
        # Processing for both grayscale and color channels images.
        if choice == 0:
            self.processed_img = self.original_img.copy()
        elif choice == 1:
            if len(self.original_img.shape) == 2:  
                self.processed_img = cv2.equalizeHist(self.original_img)
            else:  
                channels = cv2.split(self.original_img)
                equalized_channels = [cv2.equalizeHist(channel) for channel in channels]
                self.processed_img = cv2.merge(equalized_channels)
        elif choice == 2:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            if len(self.original_img.shape) == 2:  
                self.processed_img = clahe.apply(self.original_img)
            else:  
                channels = cv2.split(self.original_img)
                clahe_channels = [clahe.apply(channel) for channel in channels]
                self.processed_img = cv2.merge(clahe_channels)
        self.redraw_image()

    def redraw_image(self):
        """ Redraw the image with point and refresh the display """
        img_with_points = self.processed_img.copy()
        if self.points['grid_point']:
            cv2.drawMarker(img_with_points, self.points['grid_point'], (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)
        if self.points['bottom_point']:
            cv2.drawMarker(img_with_points, self.points['bottom_point'], (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)
        if self.points['right_point']:
            cv2.drawMarker(img_with_points, self.points['right_point'], (255, 0, 0), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)
        cv2.imshow('Image', img_with_points)
        cv2.waitKey(1)

    def on_mouse_click(self, event, x, y, flags, param):
        """ Handle mouse click events to select points """
        if event == cv2.EVENT_LBUTTONDOWN and self.point_mode:
            self.points[self.point_mode] = (x, y)
            self.redraw_image()

    def reset_point(self, point_type):
        """ Reset the selected point and refresh the image display """
        self.points[point_type] = None
        self.redraw_image()

    def create_widgets(self, prompt):
        """ Primary Construction of the GUI, including order, and functionality. """
        
        # Title & Comment Label Textbox
        Label(self, text=prompt).pack(pady=10)
        self.entry_var = StringVar(self)
        Entry(self, textvariable=self.entry_var, width=50).pack(pady=10, padx=10)

        # Confidence Levels Option
        confidence_frame = Frame(self)
        confidence_frame.pack(fill='x', padx=15, pady=5)
        self.confidence_var = IntVar()
        Label(confidence_frame, text="Labeling Confidence (Least - Most):").pack(side='left', padx=(10, 2))
        Radiobutton(confidence_frame, text="1", variable=self.confidence_var, value=1).pack(side='left')
        Radiobutton(confidence_frame, text="2", variable=self.confidence_var, value=2).pack(side='left')
        Radiobutton(confidence_frame, text="3", variable=self.confidence_var, value=3).pack(side='left')
 
        # Ionogram Present Option
        ionogram_frame = Frame(self)
        ionogram_frame.pack(fill='x', padx=15, pady=5)
        self.ionogram_var = IntVar()
        self.ionogram_var.set(0)
        Label(ionogram_frame, text="Ionogram Present:").pack(side='left', padx=(10, 2))
        Radiobutton(ionogram_frame, text="Yes", variable=self.ionogram_var, value=1).pack(side='left')
        Radiobutton(ionogram_frame, text="No", variable=self.ionogram_var, value=0).pack(side='left')
    
        # Film Type Option
        film_type_frame = Frame(self)
        film_type_frame.pack(fill='x', padx=15, pady=5)
        self.film_type_var = IntVar()
        self.film_type_var.set(1)
        Label(film_type_frame, text="Film Type:").pack(side='left', padx=(10, 2))
        Radiobutton(film_type_frame, text="White", variable=self.film_type_var, value=1).pack(side='left')
        Radiobutton(film_type_frame, text="Gray", variable=self.film_type_var, value=2).pack(side='left')
        Radiobutton(film_type_frame, text="Dark Gray", variable=self.film_type_var, value=3).pack(side='left')

        # Textboxes for Horizontal & Vertical Lines
        line_frame = Frame(self)
        line_frame.pack(fill='x', padx=15, pady=5)
    
        Label(line_frame, text="Number of Vertical Lines:").pack(side='left', padx=(10, 2))
        self.vert_lines_var = StringVar(self)
        Entry(line_frame, textvariable=self.vert_lines_var, width=10).pack(side='left', padx=(5, 20))
    
        Label(line_frame, text="Number of Horizontal Lines:").pack(side='left', padx=(10, 2))
        self.horiz_lines_var = StringVar(self)
        Entry(line_frame, textvariable=self.horiz_lines_var, width=10).pack(side='left', padx=(5, 20))

        # Grid for Points Selection
        control_frame = Frame(self)
        control_frame.pack(fill='x', padx=15, pady=5)

        for label, point_type in [("Top-Left Grid Point", 'grid_point'),
                                  ("Bottom Ionogram Point", 'bottom_point'),
                                  ("Rightmost Ionogram Point", 'right_point')]:
            button_frame = Frame(control_frame)
            button_frame.pack(fill='x', padx=15, pady=5)
            Button(button_frame, text=f"Select {label}", command=lambda pt=point_type: self.enable_point_selection(pt)).pack(side='left', padx=(20, 10), pady=10)
            Button(button_frame, text=f"Reset {label}", command=lambda pt=point_type: self.reset_point(pt)).pack(side='left', padx=(10, 20), pady=10)
    
        # Grid for Histogram Adjustment & Contrast Slider
        process_frame = Frame(self)
        process_frame.pack(fill='x', padx=15, pady=5)

        self.process_var = IntVar()
        self.process_var.set(0)
        for text, value in [("No Adjustment", 0), ("Histogram Equalization", 1), ("CLAHE", 2)]:
            Radiobutton(process_frame, text=text, variable=self.process_var, value=value, command=self.apply_image_processing).pack(anchor='w')
    
        contrast_frame = Frame(self)
        contrast_frame.pack(fill='x', padx=15, pady=5)

        self.slider = Scale(contrast_frame, from_=0, to=2.0, resolution=0.1, orient=HORIZONTAL, label="Adjust Contrast",
                            command=self.adjust_contrast)
        self.slider.set(1)
        self.slider.pack(fill='x', padx=15, pady=5)
    
        # Grid for Next & Exit 
        nav_frame = Frame(self)
        nav_frame.pack(fill='x', padx=15, pady=5)
    
        Button(nav_frame, text="Next", command=self.on_ok).pack(side='left', padx=(20, 10), pady=10)
        Button(nav_frame, text="Exit", command=self.on_exit).pack(side='right', padx=(10, 20), pady=10)

    def enable_point_selection(self, point_type):
        """ Enable point selection mode in the OpenCV window """
        self.point_mode = point_type
        cv2.setMouseCallback('Image', self.on_mouse_click, None)

    def display_image(self):
        """ Display the image initially with default settings """
        cv2.imshow('Image', self.original_img)
        cv2.waitKey(1)

    def on_ok(self):
        self.label = self.entry_var.get() if self.entry_var.get() else "NA"
        self.vertical_lines = self.vert_lines_var.get() if self.vert_lines_var.get() else "NA"
        self.horizontal_lines = self.horiz_lines_var.get() if self.horiz_lines_var.get() else "NA"
        cv2.destroyAllWindows()
        self.destroy()

    def on_exit(self):
        self.label = None
        self.user_closed = True
        cv2.destroyAllWindows()
        self.destroy()

    def show(self):
        self.wm_deiconify()
        self.wait_window()
        return self.label

def label_images_with_gui(input_path, output_path, samples, name, satellite, read_option):
    if read_option == 1:
        filenames = load_images_from_folder(input_path)
    elif read_option == 2:
        filenames = load_images_from_path(input_path, samples, output_path)
    elif read_option == 3:
        filenames = load_images_from_name(input_path, samples, output_path)
    else:
        raise ValueError("Input path must be a directory, a CSV file, or an Excel file")

    data = []

    root = Tk()
    root.withdraw()

    for filename in filenames:
        if os.path.isfile(filename):
            dialog = LabelDialog(root, "Image Label", f"(optional) Enter any comments for the image {filename}:", filename)
            label = dialog.show()
            if dialog.user_closed:
                break
    
            if label is None:
                continue
        else:
            continue

        # Structuring the data for saving. 
        data.append({
            'path': filename, 
            'satellite': satellite,
            'observer': name,
            'confidence': dialog.confidence_var.get() if dialog.confidence_var.get() != 0 else "NA",
            'comment': label, 
            'grid_point': dialog.points['grid_point'] if dialog.points['grid_point'] else "NA",
            'bottom_point': dialog.points['bottom_point'] if dialog.points['bottom_point'] else "NA",
            'right_point': dialog.points['right_point'] if dialog.points['right_point'] else "NA",
            'ionogram_present': 'Yes' if dialog.ionogram_var.get() else 'No',
            'film_type': {1: "White", 2: "Gray", 3: "Dark Gray"}[dialog.film_type_var.get()],
            'vertical_lines': dialog.vertical_lines if dialog.vertical_lines != "" else "NA",
            'horizontal_lines': dialog.horizontal_lines if dialog.horizontal_lines != "" else "NA" 
        })

    root.destroy()
    
    df = pd.DataFrame(data)
    
    if not df.empty:
        # Check against existing results and remove any duplicates
        if os.path.exists(output_path):
            existing_df = pd.read_csv(output_path, na_values=[], keep_default_na=False)
            if 'path' in existing_df.columns:
                df = df[~df['path'].isin(existing_df['path'])]
                combined_df = pd.concat([existing_df, df], ignore_index=True)
            else:
                combined_df = df
        else:
            combined_df = df
        
        try:
            combined_df.to_csv(output_path, index=False)
            print('Results successfully saved.')
        except:
            try:
                # If the file is already open on a computer//cannot be saved, it will save a local copy. 
                new_path = name + '.csv'
                combined_df.to_csv(new_path, index=False)
                print('The results have been saved locally as the file was open during saving. Please combine to full results manually.')
            except:
                print('The labeld results were unable to be saved. Please save the dataframe elsewhere.')
                
    else:
        combined_df = df
        
    return combined_df


''' Script Usage for Labelling

Provide either the path to a directory, or csv file
Directory Path: Cycles through full directory for labelling.
CSV Path: CSV with column of path; each row is the path to an image to be labelled. 

''' 

# Please provide your initials or name 
name = 'initials'

# Use either the ISIS 1 or 2 csv for general labeling, change the satellite code to corresponding satellite
input_path = r'L:\DATA\ISIS\Phase 3 - QA &Microapp& Media\result_master_ISIS1.csv'
#input_path = r'L:\DATA\ISIS\Phase 3 - QA &Microapp& Media\result_master_ISIS2.csv'
#input_path = r'L:\DATA\ISIS\Phase 3 - QA &Microapp& Media\training_data.csv'

# Please specify if it is isis-1 or isis-2 (use codes 1 or 2) for training use 3
satellite = '1'
 
# When labelling data generally please use the following
output_path = r'L:\DATA\ISIS\Phase 3 - QA &Microapp& Media\labeled_data\combined_observer_results.csv'  

# For your testing sample, please use the following (with your name replaced)
#output_path = r'L:\DATA\ISIS\Phase 3 - QA &Microapp& Media\labeled_data\your_name.csv' 

# Please specify the amount of images you would like to label (there are 20 total for training)
samples = 10

# Read options are 1: read a whole directory, 2: training data, 3: for general labeling from csv
read_option = 3

# All inputs are optional, with default values if not selected. 
data_results = label_images_with_gui(input_path, output_path, samples, name, satellite, read_option)

# If the save error message occurs, save the results to a local path to be combined. Typically occurs if the csv is opened somewhere else.
#data_results.to_csv('your_custom_location.csv', index=False)


