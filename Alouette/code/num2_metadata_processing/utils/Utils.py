from pathlib import Path
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

################################################################################
# GENERAL UTILS
################################################################################

def save_df_to_csv(df, csv_name, output_dir):
    if csv_name[-4:] != '.csv':
        csv_name += '.csv'
    
    df.to_csv(output_dir / csv_name, index=False)


################################################################################
# IMAGE COLLECTION AND PROCESSING
################################################################################

def collect_png_images(root_dir, excluded_subfolders):
    """
    Recursively collect PNG images while skipping excluded subfolders.
    """
    root_dir = Path(root_dir)

    image_paths = []
    skipped_dirs = []

    for current_dir, dirnames, filenames in os.walk(root_dir):
        current_dir = Path(current_dir)

        # Do not recurse into excluded subfolders.
        excluded_here = [d for d in dirnames if d in excluded_subfolders]
        skipped_dirs.extend([str(current_dir / d) for d in excluded_here])

        dirnames[:] = [d for d in dirnames if d not in excluded_subfolders]

        for filename in filenames:
            if filename.lower().endswith(".png"):
                image_paths.append(current_dir / filename)

    return image_paths, skipped_dirs


def read_gray(image_path):
    """
    Read image as grayscale uint8.
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    if img.ndim == 3:
        if img.shape[2] == 4:
            gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        gray = gray.astype(np.uint8)

    return gray


def save_image_inventory(image_paths, output_dir):
    inventory_df = pd.DataFrame({
        "image_path": [str(p) for p in image_paths],
        "filename": [p.name for p in image_paths],
        "parent_folder": [p.parent.name for p in image_paths],
    })
    save_df_to_csv(df=inventory_df, csv_name="Preprocessed_Images.csv", output_dir=output_dir)


################################################################################
# IMAGE VALIDATION
################################################################################


def correct_image_type(image):
    """
    Ensure image is uint8. Raise error if image is None.
    """
    if image is None:
        raise ValueError("Image is None. Cannot convert type.")
    
    if not hasattr(image, "dtype"):
        raise TypeError("Input is not a valid image (missing dtype).")

    return image.astype(np.uint8)



################################################################################
# IMAGE VISUALIZATION AND SAVING
################################################################################

def show_image_comparison(original_img, transformed_img, title1 = "Original", title2="Processed Image"):
    plt.figure(figsize=(10,4))

    # original image
    plt.subplot(1,2,1)
    plt.imshow(original_img, cmap='gray')
    plt.title(title1)
    plt.axis('off')

    # transformed image
    plt.subplot(1,2,2)
    plt.imshow(transformed_img, cmap='gray')
    plt.title(title2)
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def show_image(image, title="Image", method='cv2'):
    
    """
    Display an image using either OpenCV or matplotlib.

    Parameters:
    * image: input image (uint8)
    * title: window or plot title
    * method: display method ('cv2' or 'matplotlib')

    Returns:
    * None (displays the image)
    """

    if method == 'cv2':
        cv2.imshow(title, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif method == 'matplotlib':
        plt.figure()
        plt.imshow(image, cmap='gray')
        plt.title(title)
        plt.axis('off')
        plt.show()

    else:
        raise ValueError(f"Failed to show image due to unaccepted method type: {method}. Options include 'cv2' or 'matplotlib'")


def save_image(image, image_name, dir):
    
    """
    Save an image (uint8) to specified directory.

    Parameters:
    * image: image array (np.uint8)
    * image_name: name of the image (e.g., image001.png); defaults to .png if no extension is provided.
    * dir: destination directory (e.g., "output/")
    """

    if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_name += ".png"

    os.makedirs(dir, exist_ok=True)

    full_path = os.path.join(dir, image_name)

    successful = cv2.imwrite(full_path, image)

    if not successful:
        raise ValueError(f"Failed to save image {image_name} to {dir}")

