from pathlib import Path
import cv2

input_path = Path(
    r"L:\DATA\ISIS\2026-June-Model-Training\Cropped_Images\test\image191.png"
)

output_path = input_path.with_name(
    f"{input_path.stem}_white_background{input_path.suffix}"
)

# Load as grayscale.
image = cv2.imread(str(input_path), cv2.IMREAD_GRAYSCALE)

if image is None:
    raise FileNotFoundError(f"Could not read image: {input_path}")

# Invert colours:
# black 0 -> white 255
# white 255 -> black 0
inverted_image = cv2.bitwise_not(image)

saved = cv2.imwrite(str(output_path), inverted_image)

if not saved:
    raise IOError(f"Could not save image: {output_path}")

print(f"Saved inverted image to: {output_path}")