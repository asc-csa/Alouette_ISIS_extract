#Jeyshinee P Nov 2023 

# imports
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import tempfile

#Staring by defining a crop and copy function: 
def crop_and_copy(input_path, output_path, imageHeight):
    # Open the input image
    with Image.open(input_path) as img:
        # Get the dimensions of the image
        width, height = img.size
        plt.imshow(img)
        plt.axis("off") 
        plt.title('Original Image')
        plt.show()


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
  
# Continue then with a function to remove top noise and overwrite the image
def remove_top_bottom_noise(input_path, top_noise_height, bottom_noise_height):
    # Open the image
    with Image.open(input_path) as img:
        # Get the dimensions of the image
        width, height = img.size

        # Define the region to modify 
        region_to_modify = (0, 0, width, top_noise_height)

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

def process_middle_lines_noise(input_path, threshold, start_row, end_row):
    # Open the image
    img = Image.open(input_path)
    
    # Get the pixels
    pixels = img.load()
    width, height = img.size
    
    # Iterate through rows to process and replace colors below the threshold
    for y in range(start_row, end_row + 1):
        for x in range(width):
            r, g, b, a = pixels[x, y]
            if (r, g, b, a) < threshold:
                pixels[x, y] = (0, 0, 0, 255)

    # Iterate through all rows to process the below threshold rest pixels to black
    for y in range(18, 36):
        if y == 20 or y == 21:
           continue
        for x in range(width):
            r, g, b, a = pixels[x, y]
            if (r, g, b, a) < (32,32,32,255):
                pixels[x, y] = (0, 0, 0, 255)


    # Iterate through all rows to process the rest pixels to white
    for y in range(18, 36):
        for x in range(width):
            r, g, b, a = pixels[x, y]
            if (r, g, b, a) > (0,0,0,255):
                pixels[x, y] = (255, 255, 255, 255)

 
    # Save the modified image
    img.save(input_path.name)

#Variables for ISIS 
imageHeight = 50
top_noise_height = 18
bottom_noise_height = 15
threshold=(80, 80, 80, 255)
start_row_to_process = 20
end_row_to_process = 21

target_folder = "Target Folder"


def apply_filter(image_path, imageHeight = 50):
    #create temporary path 
    output_file_path = tempfile.NamedTemporaryFile(delete = False, suffix=".png")

    crop_and_copy(image_path, output_file_path, imageHeight)
    remove_top_bottom_noise(output_file_path, top_noise_height, bottom_noise_height)
    process_middle_lines_noise(output_file_path,threshold,start_row_to_process,end_row_to_process)

    # #Open image, get width and height 
    img  = Image.open(output_file_path.name)
    width, height = img.size
    plt.imshow(img)
    plt.axis("off") 
    plt.title('Cropped image')
    plt.show()


    #Crop leftmost side 
    left_crop_region = (0, 0, 10, height)
    left_cropped_img = img.crop(left_crop_region)
    plt.imshow(left_cropped_img)
    plt.axis("off") 
    plt.title('Left cropped image')
    plt.show()

    #Crop rightmost side 
    right_crop_region = (width-10, 0, width, height)
    right_cropped_img = img.crop(right_crop_region)
    plt.imshow(right_cropped_img)
    plt.axis("off") 
    plt.title('Right cropped image')
    plt.show()

    grayscale_left = left_cropped_img.convert("L")
    grayscale_right = right_cropped_img.convert("L")

    #Getting the minimum and maximum pixel value
    # extrema_left = grayscale_left.getextrema()
    # extrema_right = grayscale_right.getextrema()

    #print("Extrema left: " + str(extrema_left))
    #print("Extrema right: " + str(extrema_right))

    #Getting pixel
    print("Pixels on left corner: " + str(np.sum(grayscale_left)))
    print("Pixels on right corner: " + str(np.sum(grayscale_right)))

    #delete temporary file 
    output_file_path.close()
    os.unlink(output_file_path.name)




apply_filter("L:/DATA/ISIS/ISIS_101300030772/b7_R014207896/B1-34-50 ISIS A C-259/Image0261.png") #Cropped right corner
#pixels left = 7672
#pixels right = 16132
apply_filter("L:/DATA/ISIS/ISIS_101300030772/b34_R014207854/B1-35-12 ISIS A C-1876/Image0092.png") #Cropped both corners
#pixels left = 19330
#pixels right = 18499

apply_filter("L:/DATA/ISIS/ISIS_101300030772/b7_R014207896/B1-34-50 ISIS A C-260/Image0196.png") ## flagged should be cropped right only 
apply_filter("L:/DATA/ISIS/ISIS_101300030772/b18_R014207880/B1-35-32 ISIS B D-1153/Image0870.png") ## flagged 

#testing

import random
sub_directory = "L:/DATA/ISIS/ISIS_101300030772/b16_R014207847/B1-35-5 ISIS A C-1410"

random.seed(5001)
image_list = []
for root, dirs, files in os.walk(sub_directory):
    for file in files:
            image_list.append(os.path.join(root,file))  
sample_set = np.random.choice(image_list, 10, False)

for img in sample_set:
    apply_filter(img)

sample_set

# def find_rightmost_column(image_path):
#     # Load the PNG image
#     image = Image.open(image_path)

#     # Convert the image to a NumPy array
#     pixels = np.array(image)

#     # Get image width
#     width = pixels.shape[1]

#     # Display the BW image
#     plt.imshow(image)
#     plt.axis("off") 
#     plt.title('BW Image')
#     plt.show()

#     # Iterate from the rightmost column towards the left
#     for column in range(width - 1, -1, -1):
#         # Count white pixels in the current column
#         white_pixel_count = np.sum(np.all(pixels[:, column] == [255, 255, 255, 255], axis=-1))
  
#         # Check if the current column meets the criteria
#         if white_pixel_count > 1:
#             return column,image

#     # If no eligible rightmost column is found
#     print("No eligible column found - error")
#     return None