from __future__ import annotations

from pathlib import Path
import random
import shutil
from typing import Any, Iterable

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import cv2
import numpy as np
import pandas as pd
import torch
from doctr.models import detection_predictor


# =============================================================================
# CONFIG
# =============================================================================

# INPUT_SPLIT_ROOT = Path(
#     r"L:\DATA\ISIS\2026-June-Model-Training\Dataset_Split"
# )

# OUTPUT_ROOT = Path(
#     r"L:\DATA\ISIS\2026-June-Model-Training\Text_Detection"
# )

# INPUT_SPLIT_ROOT = Path(
#     r"L:\DATA\ISIS\2026-June-Model-Training\Preprocessed_Images"
# )

INPUT_SPLIT_ROOT = Path(
     r"L:\DATA\ISIS\2026-June-Model-Training\Preprocessed_Images\SSA-23\23-007"
 )

OUTPUT_ROOT = Path(
    r"L:\DATA\ISIS\Imageupload_20260428_Processing\Text_Detection\23-007"
)

#SPLITS = ("train", "validation", "test") 
SPLITS = ("",) # use this if processing a batch of images


# Random test subset across all splits combined, set to None for the full dataset
RANDOM_SUBSET_SIZE = None
RANDOM_SEED = 42

#DATASET_FOLDER_NAME = "SSA-23"
DATASET_FOLDER_NAME = ""

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
}

# False protects an existing output dataset
OVERWRITE_OUTPUT = False

SAVE_PREVIEWS = True

# Keep False when only labels, previews, and CSV reports are needed
COPY_IMAGES = False

# debug output showing the vertically cropped and height-normalized image that is tiled for FAST
SAVE_NORMALIZED_DEBUG = False

# -----------------------------------------------------------------------------
# docTR / FAST detector
# -----------------------------------------------------------------------------

MODEL_ARCH = "fast_base"
PRETRAINED = True
ASSUME_STRAIGHT_PAGES = True

# Each tile has a controlled shape, but preserving aspect ratio remains useful inside docTR's own preprocessor
PRESERVE_ASPECT_RATIO = True
SYMMETRIC_PAD = True

# Number of TILES (smaller parts of the image) sent through FAST at once, not the number of images
BATCH_SIZE = 8

PREFER_GPU = True

# The previous 0.10 / 0.10 settings were very permissive and allowed weak
# noise regions to become FAST detections. Start stricter for binary images.
BIN_THRESHOLD = 0.30
BOX_THRESHOLD = 0.30

# -----------------------------------------------------------------------------
# Adaptive preprocessing for differently sized, long metadata strips
# -----------------------------------------------------------------------------

# Detect the main horizontal text band separately for every image. No image is padded to a global dataset-wide maximum size
AUTO_VERTICAL_CROP = True

# Input images are already binary after preprocessing. The script validates
# that only black and white pixels are present and does not apply Otsu again.
VALIDATE_BINARY_INPUT = True
BINARY_PIXEL_VALUES = (0, 255)

# Small connected components are ignored only when estimating the vertical
# crop. The original image itself is not modified.
CONTENT_COMPONENT_MIN_AREA = 3
CONTENT_COMPONENT_MIN_HEIGHT = 3
CONTENT_COMPONENT_MIN_HEIGHT_FRACTION = 0.08
CONTENT_COMPONENT_MAX_WIDTH_HEIGHT_RATIO = 4.0

# Long horizontal scan/film lines can contain more foreground pixels than the
# digits and previously caused the yellow crop to lock onto the bottom edge.
REMOVE_LONG_HORIZONTAL_LINES_FOR_CROP = True
HORIZONTAL_LINE_MIN_WIDTH_PIXELS = 32
HORIZONTAL_LINE_MIN_WIDTH_FRACTION = 0.03

VERTICAL_CROP_MARGIN = 4
MIN_VERTICAL_CROP_HEIGHT = 16

# Every vertical crop is resized to this height before detection. This gives short images and tall images a consistent effective digit size.
NORMALIZED_STRIP_HEIGHT = 128

# Very wide normalized strips are divided into overlapping tiles. At 512x128, docTR sees a much less extreme aspect ratio than it sees for a 2048x45 strip.
TILE_WIDTH = 512
TILE_OVERLAP = 128
TILE_BACKGROUND_VALUE = 255

# per-tile box filters measured in normalized tile pixels
MIN_TILE_BOX_WIDTH = 2
MIN_TILE_BOX_HEIGHT = 3

# Reject detections that map back to implausibly tiny boxes in the source image
MIN_ORIGINAL_BOX_WIDTH = 1
MIN_ORIGINAL_BOX_HEIGHT = 2
MIN_BOX_HEIGHT_FRACTION_OF_CROP = 0.06

# Overlapping tiles can detect the same digit twice
DUPLICATE_IOU_THRESHOLD = 0.50

# Digits should occupy one horizontal text line
FILTER_TO_DOMINANT_TEXT_LINE = True
TEXT_LINE_CENTER_TOLERANCE_FRACTION = 0.20

# Build the final green ROI from the dominant sequence of tall foreground components inside the yellow vertical crop
# much more resistant to small noise detections than using FAST boxes alone
USE_DOMINANT_FOREGROUND_SEQUENCE_ROI = True
SEQUENCE_MIN_COMPONENT_AREA = 8
SEQUENCE_MIN_COMPONENT_HEIGHT_FRACTION_OF_CROP = 0.30
SEQUENCE_MAX_COMPONENT_WIDTH_HEIGHT_RATIO = 2.50
# A num2 metadata row normally contains 15 digits. Requiring at least 11
# character-like components prevents an isolated half-row from being accepted
# as the final ROI when the row is accidentally split at a large field gap.
SEQUENCE_MIN_COMPONENTS = 11
SEQUENCE_EXPECTED_DIGIT_COUNT = 15

# Gaps between metadata fields are substantially wider than gaps between the
# digits inside a field. The previous 3.0 multiplier was just small enough to
# split valid rows (for example, a 61 px gap with a 20 px median digit height).
# A 6.0 multiplier bridges field spacing while still rejecting distant noise.
SEQUENCE_MAX_GAP_HEIGHT_MULTIPLIER = 6.0
SEQUENCE_MIN_GAP_PIXELS = 20
SEQUENCE_MIN_SPAN_HEIGHT_MULTIPLIER = 10.0
SEQUENCE_FAST_BOX_MIN_HORIZONTAL_OVERLAP = 0.20
SEQUENCE_HORIZONTAL_PAD_HEIGHT_MULTIPLIER = 0.45

# Isolated detections far from the main number sequence are usually noise
FILTER_TO_DOMINANT_HORIZONTAL_CLUSTER = True
MIN_HORIZONTAL_CLUSTER_GAP_PIXELS = 30
HORIZONTAL_CLUSTER_GAP_HEIGHT_MULTIPLIER = 8.0

# FAST can miss an edge digit while detecting nearby digits
# The final green ROI can be expanded to adjacent foreground components that look tall enough to be part of the same number sequence.
EXPAND_ROI_USING_FOREGROUND = True
FOREGROUND_EXPANSION_MIN_AREA = 3
FOREGROUND_MIN_HEIGHT_FRACTION_OF_MODEL = 0.35
FOREGROUND_EXPANSION_MIN_GAP_PIXELS = 25
FOREGROUND_EXPANSION_GAP_HEIGHT_MULTIPLIER = 7.0

# If FAST returns no boxes at all, connected components can provide a last-resort ROI proposal. The CSV marks these boxes as "foreground_fallback".
USE_FOREGROUND_FALLBACK = True
FALLBACK_COMPONENT_MIN_AREA = 3
FALLBACK_COMPONENT_MIN_HEIGHT_FRACTION_OF_CROP = 0.30
FALLBACK_MIN_COMPONENTS = 6
FALLBACK_MIN_SPAN_HEIGHT_MULTIPLIER = 6.0

# Final merged ROI padding
PAD_X = 10
PAD_Y = 6
ROI_HORIZONTAL_EXPANSION_FRACTION = 0.05

# YOLO class information
CLASS_ID = 0
CLASS_NAME = "metadata_text"


MANIFEST_COLUMNS = [
    "split",
    "relative_path",
    "source_image_path",
    "output_image_path",
    "output_label_path",
    "preview_path",
    "normalized_debug_path",
    "model_arch",
    "device",
    "image_width",
    "image_height",
    "vertical_crop_top",
    "vertical_crop_bottom",
    "vertical_crop_height",
    "normalized_width",
    "normalized_height",
    "num_tiles",
    "num_raw_doctr_boxes",
    "num_filtered_boxes",
    "used_foreground_fallback",
    "roi_source",
    "minimum_box_score",
    "maximum_box_score",
    "mean_box_score",
    "x1",
    "y1",
    "x2",
    "y2",
    "bbox_width",
    "bbox_height",
    "yolo_center_x",
    "yolo_center_y",
    "yolo_width",
    "yolo_height",
]

BOX_COLUMNS = [
    "split",
    "relative_path",
    "box_index",
    "detection_source",
    "detector_class_name",
    "tile_index",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "normalized_x1",
    "normalized_y1",
    "normalized_x2",
    "normalized_y2",
]

FAILED_COLUMNS = [
    "split",
    "relative_path",
    "image_path",
    "reason",
]


# =============================================================================
# OUTPUT-DIRECTORY HANDLING
# =============================================================================


def prepare_output_directory(
    output_root: Path,
    overwrite: bool,
) -> None:
    if output_root.exists() and any(output_root.iterdir()):
        if not overwrite:
            raise FileExistsError(
                "The output folder already exists and is not empty:\n"
                f"{output_root}\n\n"
                "Set OVERWRITE_OUTPUT = True only when you intentionally "
                "want to delete and rebuild it."
            )

        shutil.rmtree(output_root)

    output_root.mkdir(parents=True, exist_ok=True)


# =============================================================================
# INPUT-DATASET DISCOVERY
# =============================================================================


def get_split_image_root(split_name: str) -> Path:
    split_root = INPUT_SPLIT_ROOT / split_name

    if not split_root.exists():
        raise FileNotFoundError(
            f"Split folder does not exist: {split_root}"
        )

    expected_root = split_root / DATASET_FOLDER_NAME

    if expected_root.exists() and expected_root.is_dir():
        return expected_root

    candidate_directories = [
        path
        for path in split_root.iterdir()
        if path.is_dir()
    ]

    if len(candidate_directories) == 1:
        return candidate_directories[0]

    raise FileNotFoundError(
        f"Could not identify the image folder for split '{split_name}'.\n"
        f"Expected: {expected_root}"
    )


def iter_images(root: Path) -> Iterable[Path]:
    for image_path in sorted(root.rglob("*")):
        if (
            image_path.is_file()
            and image_path.suffix.lower() in IMAGE_EXTENSIONS
        ):
            yield image_path


def chunk_list(
    items: list[Any],
    chunk_size: int,
) -> Iterable[list[Any]]:
    if chunk_size <= 0:
        raise ValueError("BATCH_SIZE must be greater than zero.")

    for start_index in range(0, len(items), chunk_size):
        yield items[start_index:start_index + chunk_size]


def select_random_subset_by_split() -> dict[str, list[Path]]:
    """Select RANDOM_SUBSET_SIZE images across all splits combined."""
    all_records: list[tuple[str, Path]] = []

    for split_name in SPLITS:
        input_image_root = get_split_image_root(split_name)
        split_paths = list(iter_images(input_image_root))
        all_records.extend(
            (split_name, image_path)
            for image_path in split_paths
        )
        print(f"Available images in '{split_name}': {len(split_paths)}")

    total_available = len(all_records)
    if total_available == 0:
        raise ValueError("No input images were found.")

    if RANDOM_SUBSET_SIZE is None:
        selected_records = all_records
        print(f"Processing all {total_available} available images.")
    else:
        if RANDOM_SUBSET_SIZE <= 0:
            raise ValueError(
                "RANDOM_SUBSET_SIZE must be greater than zero or None."
            )

        sample_size = min(RANDOM_SUBSET_SIZE, total_available)
        random_generator = random.Random(RANDOM_SEED)
        selected_records = random_generator.sample(
            all_records,
            sample_size,
        )
        print(
            f"Selected {sample_size} random images from "
            f"{total_available} using seed {RANDOM_SEED}."
        )

    selected_by_split: dict[str, list[Path]] = {
        split_name: []
        for split_name in SPLITS
    }

    for split_name, image_path in selected_records:
        selected_by_split[split_name].append(image_path)

    for split_name in SPLITS:
        selected_by_split[split_name].sort()
        print(
            f"Selected images in '{split_name}': "
            f"{len(selected_by_split[split_name])}"
        )

    return selected_by_split


# =============================================================================
# IMAGE READING AND ADAPTIVE PREPROCESSING
# =============================================================================


def read_image(
    image_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return:
      1. original OpenCV image;
      2. RGB uint8 image for docTR;
      3. grayscale uint8 image for crop/foreground analysis.
    """
    original_image = cv2.imread(
        str(image_path),
        cv2.IMREAD_UNCHANGED,
    )

    if original_image is None:
        raise ValueError(
            f"OpenCV could not read image: {image_path}"
        )

    if original_image.dtype != np.uint8:
        original_image = cv2.normalize(
            original_image,
            None,
            alpha=0,
            beta=255,
            norm_type=cv2.NORM_MINMAX,
        ).astype(np.uint8)

    if original_image.ndim == 2:
        gray_image = original_image
        rgb_image = cv2.cvtColor(
            original_image,
            cv2.COLOR_GRAY2RGB,
        )

    elif (
        original_image.ndim == 3
        and original_image.shape[2] == 4
    ):
        gray_image = cv2.cvtColor(
            original_image,
            cv2.COLOR_BGRA2GRAY,
        )
        rgb_image = cv2.cvtColor(
            original_image,
            cv2.COLOR_BGRA2RGB,
        )

    elif (
        original_image.ndim == 3
        and original_image.shape[2] == 3
    ):
        gray_image = cv2.cvtColor(
            original_image,
            cv2.COLOR_BGR2GRAY,
        )
        rgb_image = cv2.cvtColor(
            original_image,
            cv2.COLOR_BGR2RGB,
        )

    else:
        raise ValueError(
            "Unsupported image shape "
            f"{original_image.shape} for {image_path}"
        )

    return original_image, rgb_image, gray_image


def make_foreground_mask(
    gray_image: np.ndarray,
) -> np.ndarray:
    """
    Convert an already-binary image into a foreground mask where text/noise is
    255 and the dominant background is 0.

    No thresholding is performed here. In particular, Otsu is not applied a
    second time because the input images were already binarized upstream.
    """
    unique_values = np.unique(gray_image)

    if VALIDATE_BINARY_INPUT and not np.all(
        np.isin(unique_values, BINARY_PIXEL_VALUES)
    ):
        raise ValueError(
            "Expected an already-binary image containing only pixel values "
            f"{BINARY_PIXEL_VALUES}, but found {unique_values.tolist()}."
        )

    black_pixels = int(np.count_nonzero(gray_image == 0))
    white_pixels = int(np.count_nonzero(gray_image == 255))

    # The background should occupy most of the image. Choosing the minority
    # value as foreground supports both black-on-white and white-on-black input.
    foreground_value = 0 if white_pixels >= black_pixels else 255

    return np.where(
        gray_image == foreground_value,
        255,
        0,
    ).astype(np.uint8)


def remove_long_horizontal_lines(
    foreground_mask: np.ndarray,
) -> np.ndarray:
    """
    Remove long horizontal rules only from the mask used to locate the text
    band. The original image and the foreground mask used later for ROI growth
    are not changed.
    """
    if not REMOVE_LONG_HORIZONTAL_LINES_FOR_CROP:
        return foreground_mask.copy()

    image_width = foreground_mask.shape[1]
    kernel_width = max(
        HORIZONTAL_LINE_MIN_WIDTH_PIXELS,
        int(round(
            image_width * HORIZONTAL_LINE_MIN_WIDTH_FRACTION
        )),
    )
    kernel_width = min(kernel_width, image_width)

    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (kernel_width, 1),
    )

    horizontal_lines = cv2.morphologyEx(
        foreground_mask,
        cv2.MORPH_OPEN,
        horizontal_kernel,
    )

    return cv2.subtract(
        foreground_mask,
        horizontal_lines,
    )


def build_vertical_crop_candidate_mask(
    foreground_mask: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, int]]]:
    """
    Keep character-like connected components for vertical crop estimation.

    This rejects long thin film/scan lines and tiny specks before selecting the
    dominant horizontal band.
    """
    image_height = foreground_mask.shape[0]
    search_mask = remove_long_horizontal_lines(foreground_mask)

    number_of_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            search_mask,
            connectivity=8,
        )
    )

    minimum_height = max(
        CONTENT_COMPONENT_MIN_HEIGHT,
        int(np.ceil(
            image_height
            * CONTENT_COMPONENT_MIN_HEIGHT_FRACTION
        )),
    )

    candidate_mask = np.zeros_like(search_mask)
    components: list[dict[str, int]] = []

    for label_index in range(1, number_of_labels):
        x = int(stats[label_index, cv2.CC_STAT_LEFT])
        y = int(stats[label_index, cv2.CC_STAT_TOP])
        width = int(stats[label_index, cv2.CC_STAT_WIDTH])
        height = int(stats[label_index, cv2.CC_STAT_HEIGHT])
        area = int(stats[label_index, cv2.CC_STAT_AREA])

        if area < CONTENT_COMPONENT_MIN_AREA:
            continue

        if height < minimum_height:
            continue

        if width <= 0 or height <= 0:
            continue

        if (
            width / float(height)
            > CONTENT_COMPONENT_MAX_WIDTH_HEIGHT_RATIO
        ):
            continue

        candidate_mask[labels == label_index] = 255
        components.append({
            "x1": x,
            "y1": y,
            "x2": x + width,
            "y2": y + height,
            "width": width,
            "height": height,
            "area": area,
        })

    return candidate_mask, components


def filter_components(
    binary_mask: np.ndarray,
    min_area: int,
    min_height: int,
) -> tuple[np.ndarray, list[dict[str, int]]]:
    """
    Keep connected components that satisfy the supplied area/height limits.
    """
    number_of_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_mask,
        connectivity=8,
    )

    cleaned_mask = np.zeros_like(binary_mask)
    components: list[dict[str, int]] = []

    for label_index in range(1, number_of_labels):
        x = int(stats[label_index, cv2.CC_STAT_LEFT])
        y = int(stats[label_index, cv2.CC_STAT_TOP])
        width = int(stats[label_index, cv2.CC_STAT_WIDTH])
        height = int(stats[label_index, cv2.CC_STAT_HEIGHT])
        area = int(stats[label_index, cv2.CC_STAT_AREA])

        if area < min_area or height < min_height:
            continue

        cleaned_mask[labels == label_index] = 255

        components.append({
            "x1": x,
            "y1": y,
            "x2": x + width,
            "y2": y + height,
            "width": width,
            "height": height,
            "area": area,
        })

    return cleaned_mask, components


def find_true_runs(active_flags: np.ndarray) -> list[tuple[int, int]]:
    """
    Return inclusive-exclusive runs where active_flags is True.
    """
    runs: list[tuple[int, int]] = []
    run_start: int | None = None

    for index, is_active in enumerate(active_flags.tolist()):
        if is_active and run_start is None:
            run_start = index
        elif not is_active and run_start is not None:
            runs.append((run_start, index))
            run_start = None

    if run_start is not None:
        runs.append((run_start, len(active_flags)))

    return runs


def enforce_minimum_vertical_crop_height(
    crop_top: int,
    crop_bottom: int,
    image_height: int,
) -> tuple[int, int]:
    current_height = crop_bottom - crop_top

    if current_height >= MIN_VERTICAL_CROP_HEIGHT:
        return crop_top, crop_bottom

    center = (crop_top + crop_bottom) / 2.0
    half_height = MIN_VERTICAL_CROP_HEIGHT / 2.0

    new_top = int(np.floor(center - half_height))
    new_bottom = new_top + MIN_VERTICAL_CROP_HEIGHT

    if new_top < 0:
        new_top = 0
        new_bottom = min(image_height, MIN_VERTICAL_CROP_HEIGHT)

    if new_bottom > image_height:
        new_bottom = image_height
        new_top = max(0, image_height - MIN_VERTICAL_CROP_HEIGHT)

    return new_top, new_bottom


def find_vertical_text_crop(
    gray_image: np.ndarray,
) -> tuple[int, int, np.ndarray]:
    """
    Find the dominant horizontal foreground band independently for each image.

    Returns:
        crop_top, crop_bottom, original foreground mask
    """
    image_height, image_width = gray_image.shape[:2]
    foreground_mask = make_foreground_mask(gray_image)

    if not AUTO_VERTICAL_CROP:
        return 0, image_height, foreground_mask

    cleaned_mask, candidate_components = (
        build_vertical_crop_candidate_mask(
            foreground_mask
        )
    )

    if not np.any(cleaned_mask):
        return 0, image_height, foreground_mask

    # A small dilation makes broken digit strokes contribute to one row band.
    band_mask = cv2.dilate(
        cleaned_mask,
        np.ones((3, 3), dtype=np.uint8),
        iterations=1,
    )

    row_counts = np.count_nonzero(band_mask, axis=1)

    # The threshold scales mildly with width while remaining permissive for
    # narrow strings.
    minimum_active_pixels = max(
        1,
        int(round(image_width * 0.0005)),
    )

    active_rows = row_counts >= minimum_active_pixels
    runs = find_true_runs(active_rows)

    if not runs:
        return 0, image_height, foreground_mask

    def vertical_run_score(
        run: tuple[int, int],
    ) -> tuple[int, int, int]:
        run_start, run_end = run
        overlapping_components = [
            component
            for component in candidate_components
            if (
                component["y2"] > run_start
                and component["y1"] < run_end
            )
        ]

        if not overlapping_components:
            return (0, 0, 0)

        horizontal_span = (
            max(
                component["x2"]
                for component in overlapping_components
            )
            - min(
                component["x1"]
                for component in overlapping_components
            )
        )
        foreground_area = int(sum(
            component["area"]
            for component in overlapping_components
        ))

        # A true metadata line normally contains many similarly tall
        # components spread across a large horizontal span. Component count is
        # therefore more reliable than raw foreground pixels, which strongly
        # favored long scan lines in the previous implementation.
        return (
            len(overlapping_components),
            horizontal_span,
            foreground_area,
        )

    best_start, best_end = max(
        runs,
        key=vertical_run_score,
    )

    crop_top = max(0, best_start - VERTICAL_CROP_MARGIN)
    crop_bottom = min(
        image_height,
        best_end + VERTICAL_CROP_MARGIN,
    )

    crop_top, crop_bottom = enforce_minimum_vertical_crop_height(
        crop_top=crop_top,
        crop_bottom=crop_bottom,
        image_height=image_height,
    )

    if crop_bottom <= crop_top:
        return 0, image_height, foreground_mask

    return crop_top, crop_bottom, foreground_mask


def normalize_vertical_crop(
    rgb_image: np.ndarray,
    crop_top: int,
    crop_bottom: int,
) -> tuple[np.ndarray, float, float]:
    crop = rgb_image[crop_top:crop_bottom, :]

    if crop.size == 0:
        raise ValueError(
            "Vertical crop is empty: "
            f"top={crop_top}, bottom={crop_bottom}"
        )

    crop_height, crop_width = crop.shape[:2]

    target_height = NORMALIZED_STRIP_HEIGHT
    target_width = max(
        1,
        int(round(crop_width * target_height / crop_height)),
    )

    interpolation = (
        cv2.INTER_CUBIC
        if target_height > crop_height
        else cv2.INTER_AREA
    )

    normalized_image = cv2.resize(
        crop,
        (target_width, target_height),
        interpolation=interpolation,
    )

    scale_x = target_width / crop_width
    scale_y = target_height / crop_height

    return normalized_image, scale_x, scale_y


def calculate_tile_starts(
    normalized_width: int,
) -> list[int]:
    if TILE_WIDTH <= 0:
        raise ValueError("TILE_WIDTH must be greater than zero.")

    if TILE_OVERLAP < 0 or TILE_OVERLAP >= TILE_WIDTH:
        raise ValueError(
            "TILE_OVERLAP must be >= 0 and smaller than TILE_WIDTH."
        )

    if normalized_width <= TILE_WIDTH:
        return [0]

    step = TILE_WIDTH - TILE_OVERLAP
    starts = list(
        range(
            0,
            normalized_width - TILE_WIDTH + 1,
            step,
        )
    )

    final_start = normalized_width - TILE_WIDTH

    if starts[-1] != final_start:
        starts.append(final_start)

    return starts


def create_tiles(
    normalized_image: np.ndarray,
) -> list[dict[str, Any]]:
    normalized_height, normalized_width = normalized_image.shape[:2]

    if normalized_height != NORMALIZED_STRIP_HEIGHT:
        raise ValueError(
            "Unexpected normalized height: "
            f"{normalized_height}"
        )

    tile_records: list[dict[str, Any]] = []

    for tile_index, x_start in enumerate(
        calculate_tile_starts(normalized_width)
    ):
        x_end = min(
            normalized_width,
            x_start + TILE_WIDTH,
        )

        valid_width = x_end - x_start

        tile_image = np.full(
            (
                NORMALIZED_STRIP_HEIGHT,
                TILE_WIDTH,
                3,
            ),
            TILE_BACKGROUND_VALUE,
            dtype=np.uint8,
        )

        tile_image[:, :valid_width] = normalized_image[:, x_start:x_end]

        tile_records.append({
            "tile_index": tile_index,
            "x_start": x_start,
            "x_end": x_end,
            "valid_width": valid_width,
            "image": tile_image,
        })

    return tile_records


# =============================================================================
# DOCTR FAST DETECTION
# =============================================================================


def choose_device() -> torch.device:
    if PREFER_GPU and torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def create_doctr_detector(
    device: torch.device,
):
    detector = detection_predictor(
        arch=MODEL_ARCH,
        pretrained=PRETRAINED,
        assume_straight_pages=ASSUME_STRAIGHT_PAGES,
        preserve_aspect_ratio=PRESERVE_ASPECT_RATIO,
        symmetric_pad=SYMMETRIC_PAD,
        batch_size=BATCH_SIZE,
    )

    detector.model.postprocessor.bin_thresh = BIN_THRESHOLD
    detector.model.postprocessor.box_thresh = BOX_THRESHOLD

    detector = detector.to(device)
    detector.eval()

    return detector


def run_doctr_detection(
    detector,
    images: list[np.ndarray],
) -> list[dict[str, np.ndarray]]:
    predictions = detector(images)

    if len(predictions) != len(images):
        raise RuntimeError(
            "docTR returned a different number of prediction groups than "
            "input tile images."
        )

    return predictions


def detect_tiles_with_fallback(
    detector,
    tile_records: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, np.ndarray]]]:
    """
    Run the tiles in batches. A failed batch is retried tile-by-tile.
    """
    successful_pairs: list[
        tuple[dict[str, Any], dict[str, np.ndarray]]
    ] = []

    for tile_batch in chunk_list(tile_records, BATCH_SIZE):
        tile_images = [
            tile_record["image"]
            for tile_record in tile_batch
        ]

        try:
            predictions = run_doctr_detection(
                detector=detector,
                images=tile_images,
            )

            successful_pairs.extend(
                zip(tile_batch, predictions)
            )

        except Exception as batch_error:
            print(
                "    Tile batch inference failed; retrying tiles "
                f"individually. Reason: {batch_error}"
            )

            for tile_record in tile_batch:
                try:
                    prediction = run_doctr_detection(
                        detector=detector,
                        images=[tile_record["image"]],
                    )[0]

                    successful_pairs.append(
                        (tile_record, prediction)
                    )

                except Exception as tile_error:
                    print(
                        "    Tile "
                        f"{tile_record['tile_index']} failed: {tile_error}"
                    )

    return successful_pairs


def convert_tile_prediction_to_original_boxes(
    prediction: dict[str, np.ndarray],
    tile_record: dict[str, Any],
    image_width: int,
    image_height: int,
    crop_top: int,
    crop_bottom: int,
    scale_x: float,
    scale_y: float,
) -> list[dict[str, Any]]:
    converted_boxes: list[dict[str, Any]] = []
    crop_height = crop_bottom - crop_top

    minimum_original_height = max(
        MIN_ORIGINAL_BOX_HEIGHT,
        int(np.ceil(
            crop_height * MIN_BOX_HEIGHT_FRACTION_OF_CROP
        )),
    )

    for detector_class_name, class_predictions in prediction.items():
        rows = np.asarray(
            class_predictions,
            dtype=np.float32,
        )

        if rows.size == 0:
            continue

        if rows.ndim != 2 or rows.shape[1] < 5:
            raise ValueError(
                "Unexpected docTR straight-box prediction shape: "
                f"{rows.shape}"
            )

        for row in rows:
            tile_x1 = int(np.floor(float(row[0]) * TILE_WIDTH))
            tile_y1 = int(
                np.floor(
                    float(row[1]) * NORMALIZED_STRIP_HEIGHT
                )
            )
            tile_x2 = int(np.ceil(float(row[2]) * TILE_WIDTH))
            tile_y2 = int(
                np.ceil(
                    float(row[3]) * NORMALIZED_STRIP_HEIGHT
                )
            )
            confidence = float(row[4])

            # Ignore detections that fall entirely in the white padding of the
            # final short tile.
            tile_x1 = int(np.clip(
                tile_x1,
                0,
                tile_record["valid_width"],
            ))
            tile_x2 = int(np.clip(
                tile_x2,
                0,
                tile_record["valid_width"],
            ))
            tile_y1 = int(np.clip(
                tile_y1,
                0,
                NORMALIZED_STRIP_HEIGHT,
            ))
            tile_y2 = int(np.clip(
                tile_y2,
                0,
                NORMALIZED_STRIP_HEIGHT,
            ))

            if (
                tile_x2 - tile_x1 < MIN_TILE_BOX_WIDTH
                or tile_y2 - tile_y1 < MIN_TILE_BOX_HEIGHT
            ):
                continue

            normalized_global_x1 = (
                tile_record["x_start"] + tile_x1
            )
            normalized_global_x2 = (
                tile_record["x_start"] + tile_x2
            )

            x1 = int(np.floor(
                normalized_global_x1 / scale_x
            ))
            y1 = int(np.floor(
                crop_top + tile_y1 / scale_y
            ))
            x2 = int(np.ceil(
                normalized_global_x2 / scale_x
            ))
            y2 = int(np.ceil(
                crop_top + tile_y2 / scale_y
            ))

            x1 = int(np.clip(x1, 0, image_width))
            y1 = int(np.clip(y1, 0, image_height))
            x2 = int(np.clip(x2, 0, image_width))
            y2 = int(np.clip(y2, 0, image_height))

            if (
                x2 - x1 < MIN_ORIGINAL_BOX_WIDTH
                or y2 - y1 < minimum_original_height
            ):
                continue

            converted_boxes.append({
                "detection_source": "doctr_fast",
                "detector_class_name": detector_class_name,
                "tile_index": tile_record["tile_index"],
                "confidence": confidence,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "normalized_x1": x1 / image_width,
                "normalized_y1": y1 / image_height,
                "normalized_x2": x2 / image_width,
                "normalized_y2": y2 / image_height,
            })

    return converted_boxes


# =============================================================================
# BOX FILTERING, DEDUPLICATION, AND ROI CONSTRUCTION
# =============================================================================


def box_iou(
    first_box: dict[str, Any],
    second_box: dict[str, Any],
) -> float:
    intersection_x1 = max(
        first_box["x1"],
        second_box["x1"],
    )
    intersection_y1 = max(
        first_box["y1"],
        second_box["y1"],
    )
    intersection_x2 = min(
        first_box["x2"],
        second_box["x2"],
    )
    intersection_y2 = min(
        first_box["y2"],
        second_box["y2"],
    )

    intersection_width = max(
        0,
        intersection_x2 - intersection_x1,
    )
    intersection_height = max(
        0,
        intersection_y2 - intersection_y1,
    )
    intersection_area = (
        intersection_width * intersection_height
    )

    first_area = (
        (first_box["x2"] - first_box["x1"])
        * (first_box["y2"] - first_box["y1"])
    )
    second_area = (
        (second_box["x2"] - second_box["x1"])
        * (second_box["y2"] - second_box["y1"])
    )

    union_area = first_area + second_area - intersection_area

    if union_area <= 0:
        return 0.0

    return intersection_area / union_area


def deduplicate_boxes(
    boxes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Suppress near-identical detections created by overlapping tiles.
    """
    sorted_boxes = sorted(
        boxes,
        key=lambda box: box["confidence"],
        reverse=True,
    )

    kept_boxes: list[dict[str, Any]] = []

    for candidate in sorted_boxes:
        is_duplicate = any(
            box_iou(candidate, kept) >= DUPLICATE_IOU_THRESHOLD
            for kept in kept_boxes
        )

        if not is_duplicate:
            kept_boxes.append(candidate)

    return sorted(
        kept_boxes,
        key=lambda box: (
            box["x1"],
            box["y1"],
        ),
    )


def filter_to_dominant_text_line(
    boxes: list[dict[str, Any]],
    crop_height: int,
) -> list[dict[str, Any]]:
    if (
        not FILTER_TO_DOMINANT_TEXT_LINE
        or len(boxes) <= 2
    ):
        return boxes

    centers_y = np.asarray(
        [
            (box["y1"] + box["y2"]) / 2.0
            for box in boxes
        ],
        dtype=np.float32,
    )

    median_center_y = float(np.median(centers_y))
    tolerance = max(
        2.0,
        crop_height
        * TEXT_LINE_CENTER_TOLERANCE_FRACTION,
    )

    filtered = [
        box
        for box in boxes
        if (
            abs(
                (box["y1"] + box["y2"]) / 2.0
                - median_center_y
            )
            <= tolerance
            or box["y1"] <= median_center_y <= box["y2"]
        )
    ]

    return filtered if filtered else boxes


def split_into_horizontal_clusters(
    boxes: list[dict[str, Any]],
    maximum_gap: float,
) -> list[list[dict[str, Any]]]:
    if not boxes:
        return []

    sorted_boxes = sorted(
        boxes,
        key=lambda box: box["x1"],
    )

    clusters: list[list[dict[str, Any]]] = [
        [sorted_boxes[0]]
    ]
    current_right = sorted_boxes[0]["x2"]

    for box in sorted_boxes[1:]:
        gap = box["x1"] - current_right

        if gap <= maximum_gap:
            clusters[-1].append(box)
            current_right = max(
                current_right,
                box["x2"],
            )
        else:
            clusters.append([box])
            current_right = box["x2"]

    return clusters


def filter_to_dominant_horizontal_cluster(
    boxes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if (
        not FILTER_TO_DOMINANT_HORIZONTAL_CLUSTER
        or len(boxes) <= 2
    ):
        return boxes

    median_height = float(np.median([
        box["y2"] - box["y1"]
        for box in boxes
    ]))

    maximum_gap = max(
        float(MIN_HORIZONTAL_CLUSTER_GAP_PIXELS),
        median_height
        * HORIZONTAL_CLUSTER_GAP_HEIGHT_MULTIPLIER,
    )

    clusters = split_into_horizontal_clusters(
        boxes=boxes,
        maximum_gap=maximum_gap,
    )

    def cluster_score(
        cluster: list[dict[str, Any]],
    ) -> tuple[int, float, int]:
        confidence_sum = float(sum(
            box["confidence"]
            for box in cluster
        ))
        span = (
            max(box["x2"] for box in cluster)
            - min(box["x1"] for box in cluster)
        )

        return (
            len(cluster),
            confidence_sum,
            span,
        )

    return max(
        clusters,
        key=cluster_score,
    )


def components_in_text_line(
    foreground_mask: np.ndarray,
    crop_top: int,
    crop_bottom: int,
    minimum_area: int,
    minimum_height: int,
    reference_center_y: float | None,
    reference_tolerance: float | None,
) -> list[dict[str, int]]:
    crop_mask = foreground_mask[crop_top:crop_bottom, :]

    _, components = filter_components(
        binary_mask=crop_mask,
        min_area=minimum_area,
        min_height=minimum_height,
    )

    adjusted_components: list[dict[str, int]] = []

    for component in components:
        adjusted = {
            **component,
            "y1": component["y1"] + crop_top,
            "y2": component["y2"] + crop_top,
        }

        if (
            reference_center_y is not None
            and reference_tolerance is not None
        ):
            component_center_y = (
                adjusted["y1"] + adjusted["y2"]
            ) / 2.0

            if (
                abs(
                    component_center_y
                    - reference_center_y
                )
                > reference_tolerance
                and not (
                    adjusted["y1"]
                    <= reference_center_y
                    <= adjusted["y2"]
                )
            ):
                continue

        adjusted_components.append(adjusted)

    return adjusted_components


def expand_horizontal_bounds_using_foreground(
    boxes: list[dict[str, Any]],
    foreground_mask: np.ndarray,
    crop_top: int,
    crop_bottom: int,
) -> tuple[int, int]:
    left = min(box["x1"] for box in boxes)
    right = max(box["x2"] for box in boxes)

    if not EXPAND_ROI_USING_FOREGROUND:
        return left, right

    median_box_height = float(np.median([
        box["y2"] - box["y1"]
        for box in boxes
    ]))

    reference_center_y = float(np.median([
        (box["y1"] + box["y2"]) / 2.0
        for box in boxes
    ]))

    crop_height = crop_bottom - crop_top
    reference_tolerance = max(
        2.0,
        crop_height
        * TEXT_LINE_CENTER_TOLERANCE_FRACTION,
    )

    minimum_component_height = max(
        2,
        int(np.floor(
            median_box_height
            * FOREGROUND_MIN_HEIGHT_FRACTION_OF_MODEL
        )),
    )

    components = components_in_text_line(
        foreground_mask=foreground_mask,
        crop_top=crop_top,
        crop_bottom=crop_bottom,
        minimum_area=FOREGROUND_EXPANSION_MIN_AREA,
        minimum_height=minimum_component_height,
        reference_center_y=reference_center_y,
        reference_tolerance=reference_tolerance,
    )

    maximum_gap = max(
        float(FOREGROUND_EXPANSION_MIN_GAP_PIXELS),
        median_box_height
        * FOREGROUND_EXPANSION_GAP_HEIGHT_MULTIPLIER,
    )

    # Repeatedly include adjacent components. Repetition lets the ROI walk across a sequence of missed digits while stopping at distant specks.
    changed = True

    while changed:
        changed = False

        for component in components:
            if component["x2"] < left:
                gap = left - component["x2"]

                if gap <= maximum_gap:
                    left = component["x1"]
                    changed = True

            elif component["x1"] > right:
                gap = component["x1"] - right

                if gap <= maximum_gap:
                    right = component["x2"]
                    changed = True

            else:
                new_left = min(left, component["x1"])
                new_right = max(right, component["x2"])

                if new_left != left or new_right != right:
                    left = new_left
                    right = new_right
                    changed = True

    return left, right



def find_dominant_foreground_sequence(
    foreground_mask: np.ndarray,
    crop_top: int,
    crop_bottom: int,
    image_width: int,
    image_height: int,
) -> dict[str, Any] | None:
    """
    Locate the long sequence of digit-sized foreground components inside the
    already-correct yellow vertical crop.

    Small dust particles and scan noise are rejected mainly by requiring each
    candidate component to occupy a meaningful fraction of the crop height.
    Candidate components are then grouped horizontally, and the longest
    plausible group is selected.
    """
    if not USE_DOMINANT_FOREGROUND_SEQUENCE_ROI:
        return None

    crop_height = crop_bottom - crop_top

    if crop_height <= 0:
        return None

    minimum_height = max(
        2,
        int(np.ceil(
            crop_height
            * SEQUENCE_MIN_COMPONENT_HEIGHT_FRACTION_OF_CROP
        )),
    )

    crop_mask = foreground_mask[crop_top:crop_bottom, :]

    number_of_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        crop_mask,
        connectivity=8,
    )

    components: list[dict[str, Any]] = []

    for label_index in range(1, number_of_labels):
        x = int(stats[label_index, cv2.CC_STAT_LEFT])
        y = int(stats[label_index, cv2.CC_STAT_TOP])
        width = int(stats[label_index, cv2.CC_STAT_WIDTH])
        height = int(stats[label_index, cv2.CC_STAT_HEIGHT])
        area = int(stats[label_index, cv2.CC_STAT_AREA])

        if area < SEQUENCE_MIN_COMPONENT_AREA:
            continue

        if height < minimum_height:
            continue

        if width <= 0 or height <= 0:
            continue

        # Reject very wide horizontal artifacts while retaining narrow digits
        if (
            width / float(height)
            > SEQUENCE_MAX_COMPONENT_WIDTH_HEIGHT_RATIO
        ):
            continue

        components.append({
            "x1": x,
            "y1": y + crop_top,
            "x2": x + width,
            "y2": y + crop_top + height,
            "width": width,
            "height": height,
            "area": area,
        })

    if len(components) < SEQUENCE_MIN_COMPONENTS:
        return None

    median_height = float(np.median([
        component["height"]
        for component in components
    ]))

    maximum_gap = max(
        float(SEQUENCE_MIN_GAP_PIXELS),
        median_height * SEQUENCE_MAX_GAP_HEIGHT_MULTIPLIER,
    )

    sorted_components = sorted(
        components,
        key=lambda component: component["x1"],
    )

    clusters: list[list[dict[str, Any]]] = [
        [sorted_components[0]]
    ]
    current_right = sorted_components[0]["x2"]

    for component in sorted_components[1:]:
        gap = component["x1"] - current_right

        if gap <= maximum_gap:
            clusters[-1].append(component)
            current_right = max(current_right, component["x2"])
        else:
            clusters.append([component])
            current_right = component["x2"]

    plausible_clusters: list[list[dict[str, Any]]] = []

    for cluster in clusters:
        if len(cluster) < SEQUENCE_MIN_COMPONENTS:
            continue

        span = (
            max(component["x2"] for component in cluster)
            - min(component["x1"] for component in cluster)
        )

        if (
            span
            < median_height * SEQUENCE_MIN_SPAN_HEIGHT_MULTIPLIER
        ):
            continue

        plausible_clusters.append(cluster)

    if not plausible_clusters:
        return None

    def sequence_score(
        cluster: list[dict[str, Any]],
    ) -> tuple[float, float, float, int]:
        span = float(
            max(component["x2"] for component in cluster)
            - min(component["x1"] for component in cluster)
        )
        total_area = int(sum(
            component["area"]
            for component in cluster
        ))
        count = len(cluster)

        # Prefer sequences that contain most of the expected 15-digit row.
        # Span remains important, but it should not let an 8-digit half-row win
        # over a nearly complete sequence.
        expected_coverage = min(
            count,
            SEQUENCE_EXPECTED_DIGIT_COUNT,
        ) / float(SEQUENCE_EXPECTED_DIGIT_COUNT)
        count_agreement = -abs(
            count - SEQUENCE_EXPECTED_DIGIT_COUNT
        )

        return (
            expected_coverage,
            float(count_agreement),
            span,
            total_area,
        )

    best_cluster = max(
        plausible_clusters,
        key=sequence_score,
    )

    component_x1 = min(
        component["x1"]
        for component in best_cluster
    )
    component_x2 = max(
        component["x2"]
        for component in best_cluster
    )
    component_y1 = min(
        component["y1"]
        for component in best_cluster
    )
    component_y2 = max(
        component["y2"]
        for component in best_cluster
    )

    horizontal_padding = max(
        PAD_X,
        int(round(
            median_height
            * SEQUENCE_HORIZONTAL_PAD_HEIGHT_MULTIPLIER
        )),
    )

    x1 = max(0, component_x1 - horizontal_padding)
    x2 = min(image_width, component_x2 + horizontal_padding)

    # The yellow crop is already reliable, so use it as the vertical ROI boundary rather than allowing a noisy FAST box to control y1/y2.
    y1 = max(0, crop_top - PAD_Y)
    y2 = min(image_height, crop_bottom + PAD_Y)

    if x2 <= x1 or y2 <= y1:
        return None

    return {
        "bbox": (x1, y1, x2, y2),
        "component_bbox": (
            component_x1,
            component_y1,
            component_x2,
            component_y2,
        ),
        "components": best_cluster,
        "median_component_height": median_height,
        "maximum_grouping_gap": maximum_gap,
    }


def filter_fast_boxes_to_sequence(
    boxes: list[dict[str, Any]],
    sequence: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Keep only FAST boxes that overlap the foreground digit sequence.

    This removes blue noise boxes from the preview and prevents them from
    affecting CSV box outputs.
    """
    sequence_x1, _, sequence_x2, _ = sequence["bbox"]

    kept_boxes: list[dict[str, Any]] = []

    for box in boxes:
        box_width = max(1, box["x2"] - box["x1"])

        overlap_width = max(
            0,
            min(box["x2"], sequence_x2)
            - max(box["x1"], sequence_x1),
        )

        overlap_fraction = overlap_width / box_width

        if (
            overlap_fraction
            >= SEQUENCE_FAST_BOX_MIN_HORIZONTAL_OVERLAP
        ):
            kept_boxes.append(box)

    return kept_boxes


def create_foreground_fallback_boxes(
    foreground_mask: np.ndarray,
    crop_top: int,
    crop_bottom: int,
    image_width: int,
    image_height: int,
) -> list[dict[str, Any]]:
    if not USE_FOREGROUND_FALLBACK:
        return []

    crop_height = crop_bottom - crop_top
    minimum_height = max(
        2,
        int(np.ceil(
            crop_height
            * FALLBACK_COMPONENT_MIN_HEIGHT_FRACTION_OF_CROP
        )),
    )

    components = components_in_text_line(
        foreground_mask=foreground_mask,
        crop_top=crop_top,
        crop_bottom=crop_bottom,
        minimum_area=FALLBACK_COMPONENT_MIN_AREA,
        minimum_height=minimum_height,
        reference_center_y=None,
        reference_tolerance=None,
    )

    if not components:
        return []

    median_height = float(np.median([
        component["y2"] - component["y1"]
        for component in components
    ]))

    maximum_gap = max(
        float(MIN_HORIZONTAL_CLUSTER_GAP_PIXELS),
        median_height
        * HORIZONTAL_CLUSTER_GAP_HEIGHT_MULTIPLIER,
    )

    component_boxes = [
        {
            "detection_source": "foreground_fallback",
            "detector_class_name": "foreground",
            "tile_index": -1,
            "confidence": 0.0,
            "x1": component["x1"],
            "y1": component["y1"],
            "x2": component["x2"],
            "y2": component["y2"],
            "normalized_x1": component["x1"] / image_width,
            "normalized_y1": component["y1"] / image_height,
            "normalized_x2": component["x2"] / image_width,
            "normalized_y2": component["y2"] / image_height,
        }
        for component in components
    ]

    clusters = split_into_horizontal_clusters(
        boxes=component_boxes,
        maximum_gap=maximum_gap,
    )

    plausible_clusters: list[list[dict[str, Any]]] = []

    for cluster in clusters:
        if len(cluster) < FALLBACK_MIN_COMPONENTS:
            continue

        span = (
            max(box["x2"] for box in cluster)
            - min(box["x1"] for box in cluster)
        )

        if (
            span
            < median_height
            * FALLBACK_MIN_SPAN_HEIGHT_MULTIPLIER
        ):
            continue

        plausible_clusters.append(cluster)

    if not plausible_clusters:
        return []

    # Prefer a cluster containing many components, then a large horizontal
    # span and total area.
    def fallback_cluster_score(
        cluster: list[dict[str, Any]],
    ) -> tuple[int, int, int]:
        total_area = sum(
            (box["x2"] - box["x1"])
            * (box["y2"] - box["y1"])
            for box in cluster
        )
        span = (
            max(box["x2"] for box in cluster)
            - min(box["x1"] for box in cluster)
        )

        return len(cluster), span, total_area

    return max(
        plausible_clusters,
        key=fallback_cluster_score,
    )


def merge_boxes_to_bbox(
    boxes: list[dict[str, Any]],
    foreground_mask: np.ndarray,
    crop_top: int,
    crop_bottom: int,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int] | None:
    if not boxes:
        return None

    x1, x2 = expand_horizontal_bounds_using_foreground(
        boxes=boxes,
        foreground_mask=foreground_mask,
        crop_top=crop_top,
        crop_bottom=crop_bottom,
    )

    y1 = min(box["y1"] for box in boxes)
    y2 = max(box["y2"] for box in boxes)

    detected_width = x2 - x1
    fractional_expansion = int(round(
        detected_width
        * ROI_HORIZONTAL_EXPANSION_FRACTION
    ))

    x1 = max(
        0,
        x1 - PAD_X - fractional_expansion,
    )
    y1 = max(
        0,
        y1 - PAD_Y,
    )
    x2 = min(
        image_width,
        x2 + PAD_X + fractional_expansion,
    )
    y2 = min(
        image_height,
        y2 + PAD_Y,
    )

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def bbox_to_yolo(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    box_width = x2 - x1
    box_height = y2 - y1

    center_x = x1 + box_width / 2.0
    center_y = y1 + box_height / 2.0

    return (
        center_x / image_width,
        center_y / image_height,
        box_width / image_width,
        box_height / image_height,
    )


# =============================================================================
# FILE WRITING
# =============================================================================


def copy_image_without_changes(
    source_path: Path,
    destination_path: Path,
) -> None:
    destination_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    shutil.copy2(source_path, destination_path)

# each label contains one row: 0 center_x center_y width height 
# 0 is the class ID for metadata_text
# the four coordinates are normalized from 0 to 1
def write_yolo_label(
    label_path: Path,
    class_id: int,
    yolo_bbox: tuple[float, float, float, float],
) -> None:
    label_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    center_x, center_y, width, height = yolo_bbox

    label_path.write_text(
        (
            f"{class_id} "
            f"{center_x:.6f} "
            f"{center_y:.6f} "
            f"{width:.6f} "
            f"{height:.6f}\n"
        ),
        encoding="utf-8",
    )


def create_preview_canvas(
    original_image: np.ndarray,
) -> np.ndarray:
    if original_image.ndim == 2:
        return cv2.cvtColor(
            original_image,
            cv2.COLOR_GRAY2BGR,
        )

    if (
        original_image.ndim == 3
        and original_image.shape[2] == 4
    ):
        return cv2.cvtColor(
            original_image,
            cv2.COLOR_BGRA2BGR,
        )

    return original_image.copy()


def draw_preview(
    original_image: np.ndarray,
    boxes: list[dict[str, Any]],
    merged_bbox: tuple[int, int, int, int],
    crop_top: int,
    crop_bottom: int,
    preview_path: Path,
) -> None:
    preview_image = create_preview_canvas(original_image)
    image_height, image_width = preview_image.shape[:2]

    # Thin yellow lines = per-image vertical crop used for normalization
    if crop_top > 0:
        cv2.line(
            preview_image,
            (0, crop_top),
            (max(0, image_width - 1), crop_top),
            (0, 255, 255),
            1,
        )

    if crop_bottom < image_height:
        cv2.line(
            preview_image,
            (0, max(0, crop_bottom - 1)),
            (max(0, image_width - 1), max(0, crop_bottom - 1)),
            (0, 255, 255),
            1,
        )

    # Individual FAST detections = blue
    # Foreground fallback boxes = orange
    for box in boxes:
        if box["detection_source"] == "foreground_fallback":
            box_color = (0, 165, 255)
        else:
            box_color = (255, 0, 0)

        cv2.rectangle(
            preview_image,
            (box["x1"], box["y1"]),
            (
                max(box["x1"], box["x2"] - 1),
                max(box["y1"], box["y2"] - 1),
            ),
            box_color,
            1,
        )

    # Final merged metadata ROI = green
    x1, y1, x2, y2 = merged_bbox

    cv2.rectangle(
        preview_image,
        (x1, y1),
        (
            max(x1, x2 - 1),
            max(y1, y2 - 1),
        ),
        (0, 255, 0),
        2,
    )

    preview_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not cv2.imwrite(
        str(preview_path),
        preview_image,
    ):
        raise OSError(
            f"Failed to save preview image: {preview_path}"
        )


def save_normalized_debug_image(
    normalized_image: np.ndarray,
    debug_path: Path,
) -> None:
    debug_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    bgr_image = cv2.cvtColor(
        normalized_image,
        cv2.COLOR_RGB2BGR,
    )

    if not cv2.imwrite(str(debug_path), bgr_image):
        raise OSError(
            f"Failed to save normalized debug image: {debug_path}"
        )


def add_individual_box_rows(
    split_name: str,
    relative_path: str,
    boxes: list[dict[str, Any]],
    box_rows: list[dict[str, Any]],
) -> None:
    for box_index, box in enumerate(boxes):
        box_rows.append({
            "split": split_name,
            "relative_path": relative_path,
            "box_index": box_index,
            "detection_source": box["detection_source"],
            "detector_class_name": box["detector_class_name"],
            "tile_index": box["tile_index"],
            "confidence": box["confidence"],
            "x1": box["x1"],
            "y1": box["y1"],
            "x2": box["x2"],
            "y2": box["y2"],
            "normalized_x1": box["normalized_x1"],
            "normalized_y1": box["normalized_y1"],
            "normalized_x2": box["normalized_x2"],
            "normalized_y2": box["normalized_y2"],
        })


def write_dataset_yaml(
    output_root: Path,
) -> None:
    if not COPY_IMAGES:
        return

    yaml_path = output_root / "dataset.yaml"

    yaml_path.write_text(
        (
            f"path: {output_root.as_posix()}\n"
            "train: train/images\n"
            "val: validation/images\n"
            "test: test/images\n"
            "names:\n"
            f"  {CLASS_ID}: {CLASS_NAME}\n"
        ),
        encoding="utf-8",
    )


# =============================================================================
# IMAGE AND SPLIT PROCESSING
# =============================================================================


def process_image(
    split_name: str,
    image_path: Path,
    detector,
    device: torch.device,
    manifest_rows: list[dict[str, Any]],
    box_rows: list[dict[str, Any]],
    failed_rows: list[dict[str, Any]],
) -> bool:
    split_root = INPUT_SPLIT_ROOT / split_name
    relative_from_split = image_path.relative_to(split_root)
    relative_path_string = str(relative_from_split).replace("\\", "/")

    try:
        original_image, rgb_image, gray_image = read_image(
            image_path
        )

        image_height, image_width = gray_image.shape[:2]

        crop_top, crop_bottom, foreground_mask = (
            find_vertical_text_crop(gray_image)
        )

        normalized_image, scale_x, scale_y = (
            normalize_vertical_crop(
                rgb_image=rgb_image,
                crop_top=crop_top,
                crop_bottom=crop_bottom,
            )
        )

        tile_records = create_tiles(normalized_image)

        tile_prediction_pairs = detect_tiles_with_fallback(
            detector=detector,
            tile_records=tile_records,
        )

        raw_boxes: list[dict[str, Any]] = []

        for tile_record, prediction in tile_prediction_pairs:
            raw_boxes.extend(
                convert_tile_prediction_to_original_boxes(
                    prediction=prediction,
                    tile_record=tile_record,
                    image_width=image_width,
                    image_height=image_height,
                    crop_top=crop_top,
                    crop_bottom=crop_bottom,
                    scale_x=scale_x,
                    scale_y=scale_y,
                )
            )

        filtered_boxes = deduplicate_boxes(raw_boxes)

        filtered_boxes = filter_to_dominant_text_line(
            boxes=filtered_boxes,
            crop_height=crop_bottom - crop_top,
        )

        filtered_boxes = (
            filter_to_dominant_horizontal_cluster(
                boxes=filtered_boxes,
            )
        )

        used_foreground_fallback = False
        roi_source = "doctr_fast"

        foreground_sequence = find_dominant_foreground_sequence(
            foreground_mask=foreground_mask,
            crop_top=crop_top,
            crop_bottom=crop_bottom,
            image_width=image_width,
            image_height=image_height,
        )

        if foreground_sequence is not None:
            # foreground sequence controls the final green ROI
            # FAST boxes retained only when they overlap that sequence
            filtered_boxes = filter_fast_boxes_to_sequence(
                boxes=filtered_boxes,
                sequence=foreground_sequence,
            )
            merged_bbox = foreground_sequence["bbox"]
            roi_source = "foreground_sequence"

        else:
            if not filtered_boxes:
                filtered_boxes = create_foreground_fallback_boxes(
                    foreground_mask=foreground_mask,
                    crop_top=crop_top,
                    crop_bottom=crop_bottom,
                    image_width=image_width,
                    image_height=image_height,
                )
                used_foreground_fallback = bool(filtered_boxes)

            merged_bbox = merge_boxes_to_bbox(
                boxes=filtered_boxes,
                foreground_mask=foreground_mask,
                crop_top=crop_top,
                crop_bottom=crop_bottom,
                image_width=image_width,
                image_height=image_height,
            )

            if used_foreground_fallback:
                roi_source = "foreground_fallback"

        if merged_bbox is None:
            failed_rows.append({
                "split": split_name,
                "relative_path": relative_path_string,
                "image_path": str(image_path),
                "reason": (
                    "No usable foreground sequence, FAST boxes, "
                    "or foreground fallback boxes"
                ),
            })
            return False

        x1, y1, x2, y2 = merged_bbox

        yolo_bbox = bbox_to_yolo(
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            image_width=image_width,
            image_height=image_height,
        )

        output_label_path = (
            OUTPUT_ROOT
            / split_name
            / "labels"
            / relative_from_split
        ).with_suffix(".txt")

        preview_path = (
            OUTPUT_ROOT
            / split_name
            / "previews"
            / relative_from_split
        )

        output_image_path = (
            OUTPUT_ROOT
            / split_name
            / "images"
            / relative_from_split
        )

        normalized_debug_path = (
            OUTPUT_ROOT
            / split_name
            / "normalized_debug"
            / relative_from_split
        )

        if COPY_IMAGES:
            copy_image_without_changes(
                source_path=image_path,
                destination_path=output_image_path,
            )

        write_yolo_label(
            label_path=output_label_path,
            class_id=CLASS_ID,
            yolo_bbox=yolo_bbox,
        )

        if SAVE_PREVIEWS:
            draw_preview(
                original_image=original_image,
                boxes=filtered_boxes,
                merged_bbox=merged_bbox,
                crop_top=crop_top,
                crop_bottom=crop_bottom,
                preview_path=preview_path,
            )

        if SAVE_NORMALIZED_DEBUG:
            save_normalized_debug_image(
                normalized_image=normalized_image,
                debug_path=normalized_debug_path,
            )

        add_individual_box_rows(
            split_name=split_name,
            relative_path=relative_path_string,
            boxes=filtered_boxes,
            box_rows=box_rows,
        )

        scores = np.asarray(
            [
                box["confidence"]
                for box in filtered_boxes
            ],
            dtype=np.float32,
        )

        if scores.size > 0:
            minimum_box_score = float(scores.min())
            maximum_box_score = float(scores.max())
            mean_box_score = float(scores.mean())
        else:
            minimum_box_score = np.nan
            maximum_box_score = np.nan
            mean_box_score = np.nan

        center_x, center_y, yolo_width, yolo_height = (
            yolo_bbox
        )

        manifest_rows.append({
            "split": split_name,
            "relative_path": relative_path_string,
            "source_image_path": str(image_path),
            "output_image_path": (
                str(output_image_path)
                if COPY_IMAGES
                else ""
            ),
            "output_label_path": str(output_label_path),
            "preview_path": (
                str(preview_path)
                if SAVE_PREVIEWS
                else ""
            ),
            "normalized_debug_path": (
                str(normalized_debug_path)
                if SAVE_NORMALIZED_DEBUG
                else ""
            ),
            "model_arch": MODEL_ARCH,
            "device": str(device),
            "image_width": image_width,
            "image_height": image_height,
            "vertical_crop_top": crop_top,
            "vertical_crop_bottom": crop_bottom,
            "vertical_crop_height": crop_bottom - crop_top,
            "normalized_width": normalized_image.shape[1],
            "normalized_height": normalized_image.shape[0],
            "num_tiles": len(tile_records),
            "num_raw_doctr_boxes": len(raw_boxes),
            "num_filtered_boxes": len(filtered_boxes),
            "used_foreground_fallback": used_foreground_fallback,
            "roi_source": roi_source,
            "minimum_box_score": minimum_box_score,
            "maximum_box_score": maximum_box_score,
            "mean_box_score": mean_box_score,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "bbox_width": x2 - x1,
            "bbox_height": y2 - y1,
            "yolo_center_x": center_x,
            "yolo_center_y": center_y,
            "yolo_width": yolo_width,
            "yolo_height": yolo_height,
        })

        del tile_records
        del tile_prediction_pairs

        if device.type == "cuda":
            torch.cuda.empty_cache()

        return True

    except Exception as error:
        failed_rows.append({
            "split": split_name,
            "relative_path": relative_path_string,
            "image_path": str(image_path),
            "reason": f"processing failed: {error}",
        })

        if device.type == "cuda":
            torch.cuda.empty_cache()

        return False


def process_split(
    split_name: str,
    image_paths: list[Path],
    detector,
    device: torch.device,
    manifest_rows: list[dict[str, Any]],
    box_rows: list[dict[str, Any]],
    failed_rows: list[dict[str, Any]],
) -> None:
    print(f"\nProcessing split: {split_name}")
    print(f"Selected {len(image_paths)} images for this run.")

    if not image_paths:
        print(f"No sampled images in split '{split_name}'; skipping.")
        return

    successful_count = 0

    for image_index, image_path in enumerate(
        image_paths,
        start=1,
    ):
        if process_image(
            split_name=split_name,
            image_path=image_path,
            detector=detector,
            device=device,
            manifest_rows=manifest_rows,
            box_rows=box_rows,
            failed_rows=failed_rows,
        ):
            successful_count += 1

        if (
            image_index % 25 == 0
            or image_index == len(image_paths)
        ):
            print(
                "  Successful detections so far: "
                f"{successful_count}/{image_index} attempted "
                f"({len(image_paths)} total images)"
            )

    print(
        f"Finished split '{split_name}': "
        f"{successful_count} successful detections."
    )


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    prepare_output_directory(
        output_root=OUTPUT_ROOT,
        overwrite=OVERWRITE_OUTPUT,
    )

    device = choose_device()

    print("PyTorch version:", torch.__version__)
    print("CUDA runtime:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    print("Selected device:", device)
    print("Loading pretrained docTR detector:", MODEL_ARCH)

    detector = create_doctr_detector(device)

    print("Detector loaded.")
    print("BIN_THRESHOLD:", BIN_THRESHOLD)
    print("BOX_THRESHOLD:", BOX_THRESHOLD)
    print("VALIDATE_BINARY_INPUT:", VALIDATE_BINARY_INPUT)
    print(
        "REMOVE_LONG_HORIZONTAL_LINES_FOR_CROP:",
        REMOVE_LONG_HORIZONTAL_LINES_FOR_CROP,
    )
    print("NORMALIZED_STRIP_HEIGHT:", NORMALIZED_STRIP_HEIGHT)
    print("TILE_WIDTH:", TILE_WIDTH)
    print("TILE_OVERLAP:", TILE_OVERLAP)
    print("RANDOM_SUBSET_SIZE:", RANDOM_SUBSET_SIZE)
    print("RANDOM_SEED:", RANDOM_SEED)

    manifest_rows: list[dict[str, Any]] = []
    box_rows: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []

    selected_paths_by_split = select_random_subset_by_split()

    for split_name in SPLITS:
        process_split(
            split_name=split_name,
            image_paths=selected_paths_by_split[split_name],
            detector=detector,
            device=device,
            manifest_rows=manifest_rows,
            box_rows=box_rows,
            failed_rows=failed_rows,
        )

    manifest_dataframe = pd.DataFrame(
        manifest_rows,
        columns=MANIFEST_COLUMNS,
    )

    boxes_dataframe = pd.DataFrame(
        box_rows,
        columns=BOX_COLUMNS,
    )

    failed_dataframe = pd.DataFrame(
        failed_rows,
        columns=FAILED_COLUMNS,
    )

    manifest_path = (
        OUTPUT_ROOT
        / "doctr_fast_adaptive_detection_manifest.csv"
    )
    boxes_path = (
        OUTPUT_ROOT
        / "doctr_fast_adaptive_individual_boxes.csv"
    )
    failed_path = (
        OUTPUT_ROOT
        / "failed_doctr_fast_adaptive_detections.csv"
    )

    manifest_dataframe.to_csv(
        manifest_path,
        index=False,
    )

    boxes_dataframe.to_csv(
        boxes_path,
        index=False,
    )

    failed_dataframe.to_csv(
        failed_path,
        index=False,
    )

    write_dataset_yaml(OUTPUT_ROOT)

    print("\nFinished.")
    print(f"Successful images: {len(manifest_dataframe)}")
    print(f"Failed images: {len(failed_dataframe)}")
    print(f"Individual output boxes: {len(boxes_dataframe)}")
    print(f"\nDetection outputs:\n{OUTPUT_ROOT}")
    print(f"\nManifest:\n{manifest_path}")
    print(f"\nIndividual boxes:\n{boxes_path}")
    print(f"\nFailures:\n{failed_path}")

    if COPY_IMAGES:
        print(
            f"\nYOLO configuration:\n"
            f"{OUTPUT_ROOT / 'dataset.yaml'}"
        )
    else:
        print(
            "\nCOPY_IMAGES is False, so only labels, previews, and CSV reports were saved."
        )


if __name__ == "__main__":
    main()