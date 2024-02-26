#KERAS OCR script to crop the metadata part of the ionogram, remove noise and white line
#This script also applies KERAS OCR to read the metadata and uses a recognizer trained on denoised ISIS ionograms. 

#Required imports 
import os
import tensorflow as tf
import string 
from PIL import Image
import keras_ocr
import string
import tempfile

#Loading trained recognizer model
recognizer = keras_ocr.recognition.Recognizer(alphabet= string.digits) 
recognizer.model.load_weights('U:/ISIS_Extra/Metadata_Analysis/trained_recognizer.h5')   
recognizer.compile()  
pipeline = keras_ocr.pipeline.Pipeline(recognizer=recognizer)

#Paramaters for denoising code
imageHeight = 50
top_noise_height = 10
bottom_noise_height = 10
threshold_toLine=(110, 110, 110, 255)
threshold_towhite=(0, 0, 0, 255)
threshold_toblack=(80, 80, 80, 255)
start_row_to_process = 1
end_row_to_process = 20

#Functions

def crop_and_copy(input_path, output_path, imageHeight):
    # Open the input image
    with Image.open(input_path) as img:
        # Get the dimensions of the image
        width, height = img.size
        # Define the region to crop (imageHeight pixels from the bottom)
        crop_region = (0, height - imageHeight, width, height)
        # Crop the image
        cropped_img = img.crop(crop_region)
        # Create a new image with the same size as the cropped region
        new_img = Image.new("RGBA", (width, imageHeight), (0, 0, 0, 0))
        # Paste the cropped region onto the new image
        new_img.paste(cropped_img, (0, 0))
        # Save the result to the output path
        new_img.save(output_path.name)

def remove_top_bottom_noise(input_path, top_noise_height, bottom_noise_height):
    # Open the image
    with Image.open(input_path) as img:
        # Get the dimensions of the image
        width, height = img.size
        # Create a new image with the same content as the original
        new_img = img.copy()

        # Add a black border to the top noise height
        for y in range(top_noise_height):
            for x in range(width):
                new_img.putpixel((x, y), (0, 0, 0, 255))  # Set pixel to black
        # Remove noise from the bottom
        for y in range(height - bottom_noise_height, height):
            for x in range(width):
                new_img.putpixel((x, y), (0, 0, 0, 255))  # Set pixel to black

        # Save the result, overwriting the original image
        new_img.save(input_path.name)

def process_middle_lines_noise(input_path, threshold_toline, start_row, end_row):
    # Open the image
    img = Image.open(input_path)
    
    # Get the pixels
    pixels = img.load()
    width, _ = img.size
    # Iterate through rows to process and replace colors below the threshold
    for y in range(start_row, end_row + 1):
        for x in range(width):
            r, g, b, a = pixels[x, y]
            if (r, g, b, a) < threshold_toline:
                pixels[x, y] = (0, 0, 0, 255)
    # Iterate through all rows to process the below threshold rest pixels to black
    for y in range(top_noise_height, imageHeight-bottom_noise_height):
        if y == 19 or y == 20:
           continue
        for x in range(width):
            r, g, b, a = pixels[x, y]
            if (r, g, b, a) < threshold_toblack:
                pixels[x, y] = (0, 0, 0, 255)
    # Iterate through all rows to process the rest pixels to white
    for y in range(top_noise_height, imageHeight-bottom_noise_height):
        for x in range(width):
            r, g, b, a = pixels[x, y]
            if (r, g, b, a) > threshold_towhite:
                pixels[x, y] = (255, 255, 255, 255)
    # Save the modified image
    img.save(input_path.name)

def read_image(image_path,just_digits=False):
    try: 
        #applying cropping and de-noising filters
        output_file_path = tempfile.NamedTemporaryFile(delete = False, suffix=".png")
        crop_and_copy(image_path, output_file_path, imageHeight)
        remove_top_bottom_noise(output_file_path,top_noise_height,bottom_noise_height)
        process_middle_lines_noise(output_file_path,threshold_toLine,start_row_to_process,end_row_to_process)

        #reading filtered image
        image = keras_ocr.tools.read(output_file_path.name) 
      
        #applying keras
        prediction = pipeline.recognize([image])[0]

        combined_lists = list(zip([x[1][0][0] for x in prediction], [x[0] for x in prediction]))
        sorted_lists = sorted(combined_lists, key=lambda x: x[0])
        sorted_digits = [item[1] for item in sorted_lists]

    except Exception as e:
        print('ERR:', e)

    output_file_path.close()
    os.unlink(output_file_path.name)

    return sorted_digits
