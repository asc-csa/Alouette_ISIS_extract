import os
import cv2
import matplotlib.pyplot as plt

# Disclaimer: The following code was generated using Copilot

def show_image(image, title="Image", method='cv2'):
    """
    Display an image using either OpenCV or matplotlib.
    """

    if method == 'cv2':
        cv2.imshow(title, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif method == 'matplotlib':
        plt.figure(figsize=(8, 8))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis('off')
        plt.show()

    else:
        raise ValueError(
            f"Failed to show image due to unaccepted method type: {method}. "
            "Options include 'cv2' or 'matplotlib'"
        )


def show_all_images(root_folder, method='cv2'):
    """
    Traverse all subfolders and display every image.

    Parameters:
    - root_folder: directory containing the copied subfolders
    - method: 'cv2' or 'matplotlib'
    """

    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')

    for subfolder_name in sorted(os.listdir(root_folder)):
        subfolder_path = os.path.join(root_folder, subfolder_name)

        if not os.path.isdir(subfolder_path):
            continue

        print(f"\nViewing folder: {subfolder_name}")

        for image_name in sorted(os.listdir(subfolder_path)):
            if not image_name.lower().endswith(image_extensions):
                continue

            image_path = os.path.join(subfolder_path, image_name)

            image = cv2.imread(image_path)

            if image is None:
                print(f"Could not read: {image_path}")
                continue

            show_image(
                image,
                title=f"{subfolder_name} - {image_name}",
                method=method
            )


if __name__ == "__main__":
    destination_root = r""
    

    show_all_images(destination_root, method='cv2')