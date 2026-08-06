"""
Metadata Extraction from Alouette-2 Num2 Metadata Images

This script performs metadata extraction from ionogram images using a
trained docTR CRNN MobileNetV3-Small optical character recognition (OCR)
model. Metadata is extracted from image regions identified by YOLO
object detection labels, parsed according to the Alouette-2 metadata
schema, enriched with station information, and written to a
consolidated CSV file.

Prerequisites
-------------
This script assumes that metadata images have been cropped, preprocessed, 
and assigned YOLO text-detection labels through the preceding stages of this 
metadata extraction pipeline. Running this script on raw ionogram images is 
expected to reduce OCR and metadata extraction performance.


Processing Pipeline
-------------------
1. Discover image files and corresponding YOLO label files.
2. Crop the metadata region using YOLO bounding boxes obtained from the pipeline's previous text detection stage. 
3. Resize and pad metadata crops to the model input dimensions, as defined during the model training stage. 
4. Run OCR inference using a trained CRNN MobileNetV3-Small model.
5. Extract OCR prediction confidence scores.
6. Sanitize and validate OCR predictions.
7. Parse valid OCR predictions into structured metadata fields:
   - Satellite Number
   - Station Number
   - Year
   - Day of Year
   - Hour
   - Minute
   - Second
8. Convert parsed metadata into a standardized timestamp.
9. Enrich metadata using station information from a reference CSV.
10. Record extraction failures and reasons for failure.
11. Append failed text detections and intentionally excluded images to the
    final output dataset to provide a complete inventory of all images considered
    for metadata extraction. 

Metadata Schema
---------------
Expected metadata format:

    SSTTYYDDDHHMMSS

where:

    SS    = Satellite Number
    TT    = Station Number
    YY    = Year (2 digits)
    DDD   = Day of Year (1 = Jan 1)
    HH    = Hour
    MM    = Minute
    SS    = Second

After sanitization, valid metadata must contain exactly 15 numeric
characters.

Output
------
The output CSV contains:

    - Image and file information
    - OCR predictions
    - OCR confidence scores
    - Metadata extraction status
    - Failure reasons
    - Exclusion reasons
    - Parsed metadata fields
    - Derived timestamps
    - Station information
    

Failure Categories
------------------
Examples of recorded failure reasons include:

    - ocr_contains_letters
    - invalid_detected_length_<n>
    - invalid_sanitized_length_<n>
    - detection_failure - <reason>

Excluded Images
---------------
Images specified in EXCLUDED_PATHS_CSV are not processed by the OCR
pipeline. Instead, they are appended directly to the output CSV with:

    metadata_extracted = False

and

    exclusion_reason = reason specified in EXCLUDED_PATHS_CSV

Station Information
-------------------
Station metadata is retrieved from a station information CSV and
includes:

    - Station_Location
    - Station_ID
    - Station_Latitude
    - Station_Longitude

These fields are added to successfully extracted metadata records.

Disclaimer
----------
Parts of this script were developed with the assistance of Microsoft Copilot
"""

# ============================================================
# IMPORTS
# ============================================================


from pathlib import Path
import cv2
import csv
import re
import torch
from torch.utils.data import Dataset, DataLoader
from doctr.models import crnn_mobilenet_v3_small
from datetime import date, timedelta

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# ============================================================
# CONFIG
# ============================================================

IMAGES_BASE_PATH = Path(
    r"L:\DATA\ISIS\Imageupload_20260428_Processing\02_Preprocessed_Images"
)

YOLO_LABELS_BASE_PATH = Path(
    r"L:\DATA\ISIS\Imageupload_20260428_Processing\03_Text_Detection\labels"
)

MODEL_PATH = Path(
    r"L:\DATA\ISIS\2026-June-Model-Training\Text_Recognition\parseq_finetuned_summer_2026.pt"
)

FAILED_DETECTIONS_CSV = Path(
    r"L:\DATA\ISIS\Imageupload_20260428_Processing\03_Text_Detection\failed_doctr_fast_adaptive_detections.csv"
)

EXCLUDED_PATHS_CSV = Path(
    r"L:\DATA\ISIS\Imageupload_20260428_Processing\Excluded_Paths.csv"
)

STATION_INFORMATION_CSV = Path(
    r"L:\DATA\ISIS\ISIS_Test_Metadata_Analysis\ground_stations.csv"
)

OUTPUT_CSV = Path(
    r"L:\DATA\ISIS\Imageupload_20260428_Processing\Imageupload_20260428_MASTER.csv"
)

# must be compatible with the training target shape
TARGET_HEIGHT = 32
TARGET_WIDTH = 512

BATCH_SIZE = 64
NUM_WORKERS = 4


PROCESSING_IMAGE_CLASS = 'num2' # the recognizer model expects preprocessed num2 images
FAILED_PROCESSED_IMAGE_CLASS = 'loss'

# ============================================================
# MODEL
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

torch.backends.cudnn.benchmark = True

model = crnn_mobilenet_v3_small(pretrained=False)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device,
    )
)

model = model.to(device)
model.eval()


# ============================================================
# HELPERS
# ============================================================
def remove_non_alphanumeric(text: str) -> str:
    return re.sub(r'[^a-zA-Z0-9]', '', text)

def parse_metadata(text):
    """
    Parse a validated metadata string.

    Expected format:

        0-1   satellite_number
        2-3   station_number
        4-5   year
        6-8   day_of_year
        9-10  hour
        11-12 minute
        13-14 second
    """

    satellite_number = text[0:2]
    station_number = text[2:4]
    year = text[4:6]
    day_of_year = text[6:9]
    hour = text[9:11]
    minute = text[11:13]
    second = text[13:15]

    timestamp = get_timestamp(
        num_days=day_of_year,
        year_raw=year,
        hour=hour,
        minute=minute,
        second=second,
    )
    return {
        "satellite_number": satellite_number,
        "station_number": station_number,
        "year": year,
        "day_of_year": day_of_year,
        "hour": hour,
        "minute": minute,
        "second": second,
        "Timestamp": timestamp,
    }

def load_station_information(
    station_information_path: Path = STATION_INFORMATION_CSV,
):

    lookup = {}

    try:
        f = open(
                station_information_path,
                newline="",
                encoding="cp1252",
            )
        
    except UnicodeDecodeError:
            f = open(
                station_information_path,
                newline="",
                encoding="utf-8",
                )
    with f:

        reader = csv.DictReader(f)

        for row in reader:

            try:
                station_number = int(row["Number"])
            except (ValueError, TypeError):
                continue

            lookup[station_number] = row

    return lookup

def load_excluded_images(
    excluded_paths_csv: Path,
):
    """
    Load excluded paths and expand them into image records.

    Paths may point to either:

    - folders
    - subfolders
    - individual image files

    The excluded_paths_csv is expected to contain headers 'absolute_path'
    and 'reason'.

    Returns
    -------
    list[dict]

    [
        {
            "image_path": "...",
            "reason": "..."
        }
    ]
    """

    valid_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff",
        ".webp",
    }

    excluded_images = []

    if not excluded_paths_csv.exists():
        return excluded_images

    with open(
        excluded_paths_csv,
        newline="",
        encoding="cp1252",
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            path_str = row.get(
                "absolute_path",
                "",
            ).strip()

            reason = row.get(
                "reason",
                "",
            ).strip()

            if not path_str:
                continue

            path = Path(path_str)

            # Path points to a file
            if path.is_file():

                if (
                    path.suffix.lower()
                    in valid_extensions
                ):

                    excluded_images.append(
                        {
                            "image_path":
                                str(path),
                            "reason":
                                reason,
                        }
                    )

            # Path points to a folder
            elif path.is_dir():

                for image_path in path.rglob("*"):

                    if (
                        image_path.suffix.lower()
                        in valid_extensions
                    ):

                        excluded_images.append(
                            {
                                "image_path":
                                    str(image_path),
                                "reason":
                                    reason,
                            }
                        )

    return excluded_images

def contains_letters(text):
    return bool(re.search(r"[A-Za-z]", str(text)))

def get_month_and_day(num_days: int, year: int) -> tuple[int, int]:
    """
    Convert a day-of-year value into a month and day.

    Args:
        num_days: Day number within the year (1 = Jan 1).
        year: Four-digit year.

    Returns:
        A tuple containing:
            - month (1-12)
            - day (1-31)
    """
    base_date = date(year - 1, 12, 31)
    target_date = base_date + timedelta(days=num_days)

    return target_date.month, target_date.day

def get_timestamp(
    num_days: int,
    year_raw: str,
    hour: str,
    minute: str,
    second: str,
) -> str:
    """
    Build a timestamp string from year/day-of-year and time components.

    Args:
        num_days: Day number within the year (1 = Jan 1).
        year_raw: Two-digit year (e.g. "96" -> 1996).
        hour: Hour component.
        minute: Minute component.
        second: Second component.

    Returns:
        Timestamp formatted as:
            M/D/YYYY HH:MM:SS
    """

    year = int(f"19{year_raw}")
    num_days = int(num_days)
    month, day = get_month_and_day(num_days=num_days, year=year)

    return f"{month}/{day}/{year} {hour}:{minute}:{second}"

def sanitize_and_validate_metadata_text(text):
    """
    Sanitize and validate an OCR metadata prediction.

    Rules:
    - Metadata containing letters is invalid.
    - Non-alphanumeric characters are removed.
    - The resulting metadata string must contain exactly 15 characters.

    Failure reasons:
    - ocr_contains_letters
    - invalid_detected_length_<n>
    - invalid_sanitized_length_<n>

    Returns:
        (is_valid, failure_reason, sanitized_text)
    
    Examples:
        - 102499123123456
            -> valid
    
        - 10.2499123123456
            -> valid
            -> sanitized to 102499123123456
    
        - 10A499123123456
            -> invalid
            -> ocr_contains_letters
    
        - 10249912312345
            -> invalid
            -> invalid_detected_length_14
    
        - 10.249912312345
            -> invalid
            -> sanitized to 10249912312345
            -> invalid_sanitized_length_14
    """

    text = str(text)

    if re.search(r"[A-Za-z]", text):
        return (
            False,
            "ocr_contains_letters",
            text,
        )

    sanitized_text = remove_non_alphanumeric(text)

    if len(sanitized_text) != 15:

        length_type = (
            "invalid_detected_length"
            if sanitized_text == text
            else "invalid_sanitized_length"
        )

        return (
            False,
            f"{length_type}_{len(sanitized_text)}",
            sanitized_text,
        )

    return (
        True,
        "",
        sanitized_text,
    )

# ============================================================
# IMAGE PREPARATION
# ============================================================
def crop_yolo_box(image_path, yolo_path):
    image = cv2.imread(str(image_path))

    h, w = image.shape[:2]

    with open(yolo_path) as f:
        _, cx, cy, bw, bh = map(float, f.readline().split())

    x1 = int((cx - bw / 2) * w)
    y1 = int((cy - bh / 2) * h)
    x2 = int((cx + bw / 2) * w)
    y2 = int((cy + bh / 2) * h)

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    return image[y1:y2, x1:x2]

def resize_and_pad(
    image,
    target_height=32,
    target_width=128,
    pad_value=255
):
    h, w = image.shape[:2]

    scale = min(
        target_width / w,
        target_height / h
    )

    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    image = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    pad_left = (target_width - new_w) // 2
    pad_right = target_width - new_w - pad_left

    pad_top = (target_height - new_h) // 2
    pad_bottom = target_height - new_h - pad_top

    image = cv2.copyMakeBorder(
        image,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=(pad_value, pad_value, pad_value)
    )

    return image

# ============================================================
# DATASET
# ============================================================

class OCRInferenceDataset(Dataset):

    def __init__(
        self,
        images_base_path,
        yolo_labels_base_path,
        target_height,
        target_width,
    ):

        self.images_base_path = Path(images_base_path)
        self.yolo_labels_base_path = Path(yolo_labels_base_path)

        self.target_height = target_height
        self.target_width = target_width

        self.samples = []

        valid_extensions = {
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".tif",
            ".tiff",
            ".webp",
        }

        for image_path in self.images_base_path.rglob("*"):

            if image_path.suffix.lower() not in valid_extensions:
                continue

            relative_path = image_path.relative_to(
                self.images_base_path
            )

            yolo_path = (
                self.yolo_labels_base_path
                / relative_path.with_suffix(".txt")
            )

            if not yolo_path.exists():
                continue

            parts = relative_path.parts

            split = parts[0] if len(parts) > 1 else ""

            self.samples.append(
                {
                    "image_path": image_path,
                    "relative_path": relative_path,
                    "yolo_path": yolo_path,
                    "split": split,
                }
            )

        print(
            f"Found {len(self.samples)} image/yolo pairs"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        sample = self.samples[idx]

        image_path = sample["image_path"]
        relative_path = sample["relative_path"]
        yolo_path = sample["yolo_path"]

        crop = crop_yolo_box(
            image_path,
            yolo_path,
        )

        crop = resize_and_pad(
            crop,
            self.target_height,
            self.target_width,
        )

        image = torch.from_numpy(crop).float()

        if image.ndim == 2:
            image = image.unsqueeze(-1)

        image = image.permute(2, 0, 1)

        image /= 255.0

        return {
            "image": image,

            # paths
            "image_path": str(image_path),
            "relative_path": str(relative_path),
            "relative_path_no_ext": str(
                relative_path.with_suffix("")
            ),

            "top_level_folder": (
                relative_path.parts[0]
                if len(relative_path.parts) > 1
                else ""
            ),

            "subfolder": str(
                relative_path.parent
            ),

            # filenames
            "filename": image_path.name,
            "stem": image_path.stem,

            # yolo
            "yolo_path": str(yolo_path),
            "yolo_exists": True,
        }


def collate_fn(batch):

    images = torch.stack(
        [item["image"] for item in batch]
    )

    return {
        "images": images,
        "metadata": batch,
    }

# ============================================================
# MAIN
# ============================================================

def main():
    station_lookup = load_station_information()
    dataset = OCRInferenceDataset(
        images_base_path=IMAGES_BASE_PATH,
        yolo_labels_base_path=YOLO_LABELS_BASE_PATH,
        target_height=TARGET_HEIGHT,
        target_width=TARGET_WIDTH,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=collate_fn,
    )

    fieldnames = [
        "image_path",
        "relative_path",
        "top_level_folder",
        "subfolder",
        "filename",
        "stem",
        "processed_image_class",
        "yolo_path",

        "predicted_text",
        "prediction_confidence",

        "metadata_extracted",
        "failure_reason",
        "exclusion_reason",

        "satellite_number",
        "station_number",
        "year",
        "day_of_year",
        "hour",
        "minute",
        "second",
        "Timestamp",

        "Station_Location",
        "Station_ID",
        "Station_Latitude",
        "Station_Longitude",

    ]

    # -------------- Retrieve Failed Detections --------------

    failed_detections = []

    if Path(FAILED_DETECTIONS_CSV).exists():

        try:
            f = open(
                    FAILED_DETECTIONS_CSV,
                    newline="",
                    encoding="cp1252",
                )
            
        except UnicodeDecodeError:
            f = open(
                    FAILED_DETECTIONS_CSV,
                    newline="",
                    encoding="utf-8",
                )    
    
        with f:
            reader = csv.DictReader(f)

            for row in reader:
                failed_detections.append(row)

     # -------------- Retrieve Images from Excluded Paths --------------
    
    excluded_images = load_excluded_images(
        EXCLUDED_PATHS_CSV
    )

     # -------------- Initialize the Output CSV --------------
    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="cp1252",
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        # -------------- Make Metadata OCR Predictions --------------

        with torch.no_grad():

            autocast_enabled = device.type == "cuda"

            with torch.autocast(
                device_type=device.type,
                enabled=autocast_enabled,
            ):

                for batch in dataloader:

                    images = batch["images"].to(
                        device,
                        non_blocking=True,
                    )

                    predictions = model(images)

                    predictions_list = []

                    for text, confidence in predictions["preds"]:

                        predictions_list.append(
                            {
                                "text": str(text),
                                "confidence": float(confidence),
                            }
                        )

                    for metadata, pred in zip(
                        batch["metadata"],
                        predictions_list,
                    ):

                        text = pred["text"]
                        confidence = pred["confidence"]

                        # -------------- Validate Extracted Metadata --------------

                        parsed = None
                        metadata_extracted = False
                        failure_reason = ""
                        processing_image_class = PROCESSING_IMAGE_CLASS

                        (
                            is_valid,
                            validation_failure_reason,
                            cleaned_text,
                        ) = sanitize_and_validate_metadata_text(text)

                        if is_valid:

                            parsed = parse_metadata(
                                cleaned_text
                            )
                            text = cleaned_text
                            metadata_extracted = True

                        else:

                            failure_reason = (
                                validation_failure_reason
                            )

                            processing_image_class = (
                                FAILED_PROCESSED_IMAGE_CLASS
                            )


                        # -------------- Extract Metadata from Prediction --------------
                        row = {
                            "image_path":
                                metadata["image_path"],

                            "relative_path":
                                metadata["relative_path"],

                            "top_level_folder":
                                metadata["top_level_folder"],

                            "subfolder":
                                metadata["subfolder"],

                            "filename":
                                metadata["filename"],

                            "stem":
                                metadata["stem"],

                            "processed_image_class":
                                processing_image_class,

                            "yolo_path":
                                metadata["yolo_path"],

                            "predicted_text":
                                text,

                            "prediction_confidence":
                                confidence,

                            "metadata_extracted":
                                metadata_extracted,

                            "failure_reason":
                                failure_reason,

                            "exclusion_reason":
                                "",

                            "satellite_number": "",
                            "station_number": "",
                            "year": "",
                            "day_of_year": "",
                            "hour": "",
                            "minute": "",
                            "second": "",

                        }

                        # -------------- Parse the Metadata to Extract Meaningful Information --------------

                        if parsed is not None:

                            row.update(parsed)

                            station_info = station_lookup.get(
                                int(parsed["station_number"]),
                                {}
                            )

                            row.update({
                                "Station_Location": station_info.get("Station_Location", ""),
                                "Station_ID": station_info.get("Station_ID", ""),
                                "Station_Latitude": station_info.get("Station_Latitude", ""),
                                "Station_Longitude": station_info.get("Station_Longitude", ""),
                            })

                        writer.writerow(row)

        # ====================================================
        # APPEND FAILED DETECTIONS
        # ====================================================

        for failed in failed_detections:

            image_path = failed.get("image_path", "")
            relative_path = failed.get("relative_path", "")


            writer.writerow(
                {
                    "image_path": image_path,

                    "relative_path": relative_path,

                    "top_level_folder":
                        Path(relative_path).parts[0]
                        if relative_path else "",

                    "subfolder":
                        str(Path(relative_path).parent)
                        if relative_path else "",

                    "filename":
                        Path(image_path).name,

                    "stem":
                        Path(image_path).stem,

                    "processed_image_class":
                        FAILED_PROCESSED_IMAGE_CLASS,

                    "yolo_path": "",

                    "predicted_text": "",
                    "prediction_confidence": "",

                    "metadata_extracted": False,

                    "failure_reason":
                        "detection_failure - " + failed.get(
                            "reason",
                            "failed_detection",
                        ),

                    "exclusion_reason":
                        "",

                    "satellite_number": "",
                    "station_number": "",
                    "year": "",
                    "day_of_year": "",
                    "hour": "",
                    "minute": "",
                    "second": "",

                }
            )

        # ====================================================
        # APPEND EXCLUDED IMAGES
        # ====================================================

        for excluded in excluded_images:

            image_path = excluded["image_path"]

            writer.writerow(
                {
                    "image_path":
                        image_path,

                    "relative_path": "",
                    "top_level_folder": "",
                    "subfolder": "",

                    "filename":
                        Path(image_path).name,

                    "stem":
                        Path(image_path).stem,

                    "processed_image_class":
                        FAILED_PROCESSED_IMAGE_CLASS,

                    "yolo_path": "",

                    "predicted_text": "",
                    "prediction_confidence": "",

                    "metadata_extracted":
                        False,

                    "failure_reason": "",

                    "exclusion_reason":
                        excluded["reason"],

                    "satellite_number": "",
                    "station_number": "",
                    "year": "",
                    "day_of_year": "",
                    "hour": "",
                    "minute": "",
                    "second": "",

                    "Timestamp": "",

                    "Station_Location": "",
                    "Station_ID": "",
                    "Station_Latitude": "",
                    "Station_Longitude": "",

                }
            )
            

    print(
        f"✅ Saved metadata extraction to {OUTPUT_CSV}"
    )


if __name__ == "__main__":
    main()

