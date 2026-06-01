"""
Unified Alouette/ISIS ionogram satellite classifier.

Classifies each ionogram into:
    1 -> Alouette 1
    2 -> Alouette 2
    3 -> ISIS 1
    4 -> ISIS 2

The classifier does not train a new image-level CNN because the satellite label is
already encoded as the first metadata value. Instead, it uses the same keras-ocr
approach used in the existing Alouette/ISIS scripts, plus an ensemble of metadata
crops so it can handle metadata printed at the bottom or rotated on the side.

Examples:
    python classifier.py --input_path ./Alouette --output_csv ./classifier/output/results.csv --recursive

    python classifier.py --input_path ./ISIS --output_csv ./classifier/output/isis_results.csv --recursive \
        --recognizer_weights "L:/DATA/ISIS/keras_ocr_training/ISIS_reading_final.h5"

    python .\classifier\code\classifier.py `                                                                         
     --input_path ".\classifier\multimedia" `
     --output_csv ".\classifier\output\test_predictions.csv" `
     --batch_size 1 `
     --save_debug_crops_dir ".\classifier\debug_crops"

Optional GPU environment path, matching the style of Alouette_processor2.py:
    python classifier.py --input_path ./data --output_csv results.csv --recursive \
        --gpu_env_path "U:/temp/user/python/envs/tf210/lib/site-packages/"
"""

from __future__ import annotations

import argparse
import gc
import os
import re
import string
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np
import pandas as pd
from PIL import Image


SATELLITE_CODE_TO_NAME = {
    "1": "Alouette 1",
    "2": "Alouette 2",
    "3": "ISIS 1",
    "4": "ISIS 2",
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


@dataclass
class CandidateCrop:
    """One possible metadata crop/orientation to run through OCR."""

    image: np.ndarray
    location: str
    orientation: str
    crop_name: str


@dataclass
class OCRItem:
    """Single OCR text box."""

    text: str
    digits: str
    cx: float
    cy: float
    x_min: float
    y_min: float
    x_max: float
    y_max: float


@dataclass
class ClassificationResult:
    """One CSV row per ionogram."""

    filename: str
    satellite_code: str
    satellite_name: str
    status: str
    confidence: float
    metadata_location: str
    metadata_orientation: str
    crop_name: str
    read_string: str
    classifier_method: str
    details: str
    processing_seconds: float


# ---------------------------------------------------------------------------
# Model setup: adapted from Alouette_processor2.py and 02_Metadata processing.py
# ---------------------------------------------------------------------------


def build_keras_ocr_pipeline(gpu_env_path: str = "", recognizer_weights: str = ""):
    """
    Build a keras_ocr pipeline.

    - If gpu_env_path is provided, it is inserted before tensorflow/keras_ocr are imported,
      following the pattern in Alouette_processor2.py.
    - If recognizer_weights is provided, use a digit-only recognizer, following the ISIS
      metadata OCR script pattern.
    - Otherwise, use the default keras_ocr pipeline.
    """
    if gpu_env_path:
        sys.path.insert(0, gpu_env_path)

    import tensorflow as tf  # noqa: WPS433 - intentionally lazy import
    import keras_ocr  # noqa: WPS433 - intentionally lazy import

    print("TensorFlow version:", tf.__version__)
    print("GPU devices:", tf.config.list_physical_devices("GPU"))

    if recognizer_weights:
        recognizer = keras_ocr.recognition.Recognizer(alphabet=string.digits)
        recognizer.model.load_weights(recognizer_weights)
        recognizer.compile()
        return keras_ocr.pipeline.Pipeline(recognizer=recognizer)

    return keras_ocr.pipeline.Pipeline()


# ---------------------------------------------------------------------------
# Image loading, cropping, and denoising
# ---------------------------------------------------------------------------


def read_image_rgb(image_path: Path) -> np.ndarray:
    """Read image with PIL and return RGB numpy array."""
    with Image.open(image_path) as img:
        return np.array(img.convert("RGB"))


def resize_for_ocr(image_rgb: np.ndarray, max_width: int = 1800) -> np.ndarray:
    """Downscale very large crops to keep keras-ocr memory usage reasonable."""
    height, width = image_rgb.shape[:2]
    if width <= max_width:
        return image_rgb
    scale = max_width / float(width)
    new_size = (max_width, max(1, int(height * scale)))
    return cv2.resize(image_rgb, new_size, interpolation=cv2.INTER_AREA)


def denoise_metadata_crop(
    image_rgb: np.ndarray,
    top_noise_height: int = 10,
    bottom_noise_height: int = 10,
    threshold_toblack: int = 80,
) -> np.ndarray:
    """
    Denoise a metadata strip.

    This is the in-memory version of the ISIS functions crop_and_copy,
    remove_top_bottom_noise, and process_middle_lines_noise. It keeps bright
    metadata digits/dots and suppresses darker film/grid noise.
    """
    img = image_rgb.copy()
    height = img.shape[0]

    if height > top_noise_height + bottom_noise_height:
        img[:top_noise_height, :, :] = 0
        img[height - bottom_noise_height :, :, :] = 0

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, threshold_toblack, 255, cv2.THRESH_BINARY)

    # A light morphology close helps broken OCR strokes/dots without overly merging digits.
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    return cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)


def crop_bottom(image_rgb: np.ndarray, crop_height: int) -> np.ndarray:
    """Crop bottom metadata strip, based on the ISIS script's imageHeight crop."""
    height, _ = image_rgb.shape[:2]
    crop_height = min(crop_height, height)
    return image_rgb[height - crop_height : height, :, :]


def crop_left(image_rgb: np.ndarray, crop_width: int) -> np.ndarray:
    """Crop left-side metadata strip for rotated numerical/dot metadata."""
    _, width = image_rgb.shape[:2]
    crop_width = min(crop_width, width)
    return image_rgb[:, :crop_width, :]


def crop_right(image_rgb: np.ndarray, crop_width: int) -> np.ndarray:
    """Crop right-side metadata strip as a rare fallback."""
    _, width = image_rgb.shape[:2]
    crop_width = min(crop_width, width)
    return image_rgb[:, width - crop_width : width, :]


def rotate_crop(image_rgb: np.ndarray, orientation: str) -> np.ndarray:
    """Rotate a crop to make side metadata approximately horizontal for OCR."""
    if orientation == "none":
        return image_rgb
    if orientation == "cw90":
        return cv2.rotate(image_rgb, cv2.ROTATE_90_CLOCKWISE)
    if orientation == "ccw90":
        return cv2.rotate(image_rgb, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if orientation == "180":
        return cv2.rotate(image_rgb, cv2.ROTATE_180)
    raise ValueError(f"Unsupported orientation: {orientation}")


def make_candidate_crops(
    image_rgb: np.ndarray,
    include_full_image_fallback: bool = True,
) -> list[CandidateCrop]:
    """
    Generate metadata crops for different layouts.

    Intentional ordering:
    1. bottom strips for ISIS-style/num2 metadata,
    2. left side strips rotated both ways,
    3. right side strips as fallback,
    4. full image fallback.
    """
    candidates: list[CandidateCrop] = []
    height, width = image_rgb.shape[:2]

    # Bottom metadata: ISIS code used imageHeight=50. Extra heights improve robustness.
    for crop_height in [50, 70, 90, 120]:
        if crop_height < height:
            raw = crop_bottom(image_rgb, crop_height)
            candidates.append(
                CandidateCrop(
                    image=denoise_metadata_crop(raw),
                    location="bottom",
                    orientation="none",
                    crop_name=f"bottom_{crop_height}px_denoised",
                )
            )

    # Side metadata: metadata can appear on the side and be rotated 90 degrees.
    side_widths = sorted({
        max(60, int(width * 0.08)),
        max(90, int(width * 0.12)),
        max(120, int(width * 0.18)),
    })
    for crop_width in side_widths:
        if crop_width < width:
            for side_name, crop_fn in [("left", crop_left), ("right", crop_right)]:
                side = crop_fn(image_rgb, crop_width)
                for orientation in ["cw90", "ccw90"]:
                    rotated = rotate_crop(side, orientation)
                    candidates.append(
                        CandidateCrop(
                            image=denoise_metadata_crop(rotated),
                            location=side_name,
                            orientation=orientation,
                            crop_name=f"{side_name}_{crop_width}px_{orientation}_denoised",
                        )
                    )

    if include_full_image_fallback:
        candidates.append(
            CandidateCrop(
                image=resize_for_ocr(image_rgb),
                location="full_image",
                orientation="none",
                crop_name="full_image_fallback",
            )
        )

    return candidates


# ---------------------------------------------------------------------------
# OCR parsing and classification
# ---------------------------------------------------------------------------


def normalize_ocr_text(text: str) -> str:
    """Normalize common OCR confusions and keep only digits."""
    replacements = {
        "o": "0",
        "O": "0",
        "i": "1",
        "I": "1",
        "l": "1",
        "|": "1",
        "!": "1",
        "s": "5",
        "S": "5",
        "z": "2",
        "Z": "2",
        "A": "4",
        "a": "4",
        "b": "8",
        "B": "8",
    }
    fixed = "".join(replacements.get(ch, ch) for ch in str(text))
    return re.sub(r"\D", "", fixed)


def prediction_to_items(prediction) -> list[OCRItem]:
    """Convert keras-ocr prediction boxes into sortable OCRItem records."""
    items: list[OCRItem] = []
    for text, box in prediction:
        box_arr = np.asarray(box, dtype=float)
        digits = normalize_ocr_text(text)
        if not digits:
            continue

        x_min = float(box_arr[:, 0].min())
        y_min = float(box_arr[:, 1].min())
        x_max = float(box_arr[:, 0].max())
        y_max = float(box_arr[:, 1].max())
        items.append(
            OCRItem(
                text=str(text),
                digits=digits,
                cx=float(box_arr[:, 0].mean()),
                cy=float(box_arr[:, 1].mean()),
                x_min=x_min,
                y_min=y_min,
                x_max=x_max,
                y_max=y_max,
            )
        )
    return items


def concatenate_left_to_right(items: list[OCRItem]) -> str:
    """Concatenate OCR digits from left to right, as in the existing scripts."""
    return "".join(item.digits for item in sorted(items, key=lambda item: item.cx))


def group_items_by_row(items: list[OCRItem], tolerance: float = 18.0) -> list[list[OCRItem]]:
    """
    Group OCR boxes into approximate horizontal rows.

    This is useful for rotated side metadata, where the satellite number can be
    represented by digits on the same line that must be added together.
    """
    if not items:
        return []

    rows: list[list[OCRItem]] = []
    for item in sorted(items, key=lambda it: it.cy):
        placed = False
        for row in rows:
            row_y = float(np.mean([existing.cy for existing in row]))
            if abs(item.cy - row_y) <= tolerance:
                row.append(item)
                placed = True
                break
        if not placed:
            rows.append([item])

    return [sorted(row, key=lambda it: it.cx) for row in rows]


def row_value(row: list[OCRItem]) -> Optional[int]:
    """
    Interpret one numerical metadata row.

    For numerical/punctiform-style metadata, values on the same line are summed
    (usually weights 1, 2, 4, 8). This function returns a value only if the row
    sum is a valid satellite code, 1-4.
    """
    digits = "".join(item.digits for item in row)
    values = [int(ch) for ch in digits if ch in {"1", "2", "4", "8"}]
    if not values:
        return None
    value = sum(values)
    if 1 <= value <= 4:
        return value
    return None


def infer_satellite_code(
    read_string: str,
    items: list[OCRItem],
    candidate: CandidateCrop,
) -> tuple[str, str, float]:
    """
    Infer the satellite code from OCR output.

    Returns:
        satellite_code, method_description, method_bonus_score
    """
    # Bottom/num2 case: existing code treats read_str[0:1] as Satellite_Code.
    if read_string and read_string[0] in SATELLITE_CODE_TO_NAME:
        return read_string[0], "first_digit_of_metadata_string", 40.0

    # Sometimes OCR catches a leading separator/noise before the metadata. Use the first
    # valid metadata digit as a weaker fallback.
    for index, ch in enumerate(read_string):
        if ch in SATELLITE_CODE_TO_NAME:
            return ch, f"first_valid_satellite_digit_at_position_{index}", 20.0

    # Side numerical metadata case: values on the same row may need to be added.
    rows = group_items_by_row(items)
    if rows and candidate.location in {"left", "right"}:
        # Try both extremes because the chosen 90-degree rotation determines whether the
        # satellite-number line appears at the top or bottom of the rotated crop.
        row_options = [
            ("bottom_row_sum", rows[-1]),
            ("top_row_sum", rows[0]),
        ]
        for method_name, row in row_options:
            value = row_value(row)
            if value is not None:
                return str(value), method_name, 30.0

    return "", "no_valid_satellite_code_found", 0.0


def score_candidate(
    code: str,
    read_string: str,
    method_bonus: float,
    candidate: CandidateCrop,
) -> float:
    """Heuristic confidence score for choosing the best crop/orientation."""
    score = 0.0
    digit_count = len(read_string)

    if code in SATELLITE_CODE_TO_NAME:
        score += 100.0
    score += min(digit_count, 18) * 2.0
    score += method_bonus

    # Full 15-character metadata is the strongest pattern in the existing OCR scripts.
    if digit_count == 15:
        score += 25.0
    elif 10 <= digit_count <= 18:
        score += 15.0
    elif 1 <= digit_count < 10:
        score += 4.0

    if candidate.location == "bottom":
        score += 5.0
    elif candidate.location in {"left", "right"}:
        score += 3.0

    return score


def score_to_confidence(score: float) -> float:
    """Map the heuristic score to a readable 0-1 confidence value."""
    return round(float(max(0.0, min(0.99, score / 180.0))), 3)


def run_ocr_on_candidate(pipeline, candidate: CandidateCrop) -> tuple[str, list[OCRItem]]:
    """Run keras-ocr on one candidate crop."""
    prediction = pipeline.recognize([candidate.image])[0]
    items = prediction_to_items(prediction)
    read_string = concatenate_left_to_right(items)
    return read_string, items


def classify_image(
    image_path: Path,
    pipeline,
    save_debug_crops_dir: str = "",
    include_full_image_fallback: bool = True,
    early_stop_score: float = 165.0,
) -> ClassificationResult:
    """Classify one ionogram image."""
    start = time.time()
    image_rgb = read_image_rgb(image_path)
    candidates = make_candidate_crops(
        image_rgb=image_rgb,
        include_full_image_fallback=include_full_image_fallback,
    )

    best = {
        "score": -1.0,
        "code": "",
        "read_string": "",
        "method": "no_ocr_run",
        "candidate": CandidateCrop(
            image=np.zeros((1, 1, 3), dtype=np.uint8),
            location="none",
            orientation="none",
            crop_name="none",
        ),
        "details": "",
    }

    debug_dir = Path(save_debug_crops_dir) if save_debug_crops_dir else None
    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)

    for candidate_index, candidate in enumerate(candidates):
        try:
            read_string, items = run_ocr_on_candidate(pipeline, candidate)
            code, method, method_bonus = infer_satellite_code(read_string, items, candidate)
            score = score_candidate(code, read_string, method_bonus, candidate)

            if debug_dir:
                safe_stem = re.sub(r"[^A-Za-z0-9_.-]", "_", image_path.stem)
                debug_path = debug_dir / f"{safe_stem}_{candidate_index:02d}_{candidate.crop_name}.png"
                cv2.imwrite(str(debug_path), cv2.cvtColor(candidate.image, cv2.COLOR_RGB2BGR))

            if score > float(best["score"]):
                best = {
                    "score": score,
                    "code": code,
                    "read_string": read_string,
                    "method": method,
                    "candidate": candidate,
                    "details": f"digits_read={len(read_string)}; raw_text_items={[item.text for item in items]}",
                }

            # Stop early for a highly plausible full metadata read.
            if score >= early_stop_score and len(read_string) >= 10 and code in SATELLITE_CODE_TO_NAME:
                break

        except Exception as exc:  # keep batch processing alive even if one crop fails
            details = f"OCR failed on {candidate.crop_name}: {exc}"
            if float(best["score"]) < 0:
                best["details"] = details
            continue

    candidate = best["candidate"]
    code = str(best["code"])
    satellite_name = SATELLITE_CODE_TO_NAME.get(code, "Unknown")
    status = "classified" if code in SATELLITE_CODE_TO_NAME else "unclassified"

    return ClassificationResult(
        filename=str(image_path),
        satellite_code=code,
        satellite_name=satellite_name,
        status=status,
        confidence=score_to_confidence(float(best["score"])),
        metadata_location=candidate.location,
        metadata_orientation=candidate.orientation,
        crop_name=candidate.crop_name,
        read_string=str(best["read_string"]),
        classifier_method=str(best["method"]),
        details=str(best["details"]),
        processing_seconds=round(time.time() - start, 3),
    )


# ---------------------------------------------------------------------------
# Batch/file handling
# ---------------------------------------------------------------------------


def iter_image_paths(input_path: Path, recursive: bool) -> Iterable[Path]:
    """Yield image files from a file or directory."""
    if input_path.is_file():
        if input_path.suffix.lower() in IMAGE_EXTENSIONS:
            yield input_path
        return

    pattern = "**/*" if recursive else "*"
    for path in sorted(input_path.glob(pattern)):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def classify_images(
    input_path: Path,
    output_csv: Path,
    batch_size: int,
    recursive: bool,
    gpu_env_path: str,
    recognizer_weights: str,
    save_debug_crops_dir: str,
    include_full_image_fallback: bool,
) -> pd.DataFrame:
    """Classify all images and save a CSV."""
    pipeline = build_keras_ocr_pipeline(
        gpu_env_path=gpu_env_path,
        recognizer_weights=recognizer_weights,
    )

    image_paths = list(iter_image_paths(input_path, recursive=recursive))
    if not image_paths:
        raise FileNotFoundError(f"No image files found in: {input_path}")

    print(f"Found {len(image_paths)} image(s) to classify.")
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for index, image_path in enumerate(image_paths, start=1):
        print(f"[{index}/{len(image_paths)}] Classifying {image_path}")
        result = classify_image(
            image_path=image_path,
            pipeline=pipeline,
            save_debug_crops_dir=save_debug_crops_dir,
            include_full_image_fallback=include_full_image_fallback,
        )
        rows.append(asdict(result))

        # Incremental save prevents losing all progress if a long run is interrupted.
        if len(rows) % max(1, batch_size) == 0 or index == len(image_paths):
            df_partial = pd.DataFrame(rows)
            df_partial.to_csv(output_csv, index=False)
            gc.collect()

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"Saved results to: {output_csv}")
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify Alouette/ISIS ionograms from the first metadata satellite digit.",
    )
    parser.add_argument(
        "--input_path",
        required=True,
        help="Image file or directory containing ionogram images.",
    )
    parser.add_argument(
        "--output_csv",
        default="satellite_classification_results.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search input_path recursively for images.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10,
        help="How often to save intermediate CSV progress.",
    )
    parser.add_argument(
        "--gpu_env_path",
        default="",
        help="Optional path to TensorFlow GPU environment site-packages.",
    )
    parser.add_argument(
        "--recognizer_weights",
        default="",
        help="Optional keras-ocr digit recognizer weights, e.g. ISIS_reading_final.h5.",
    )
    parser.add_argument(
        "--save_debug_crops_dir",
        default="",
        help="Optional directory to save the OCR crops used by the classifier.",
    )
    parser.add_argument(
        "--no_full_image_fallback",
        action="store_true",
        help="Disable the final full-image OCR fallback to make processing faster.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    classify_images(
        input_path=Path(args.input_path),
        output_csv=Path(args.output_csv),
        batch_size=args.batch_size,
        recursive=args.recursive,
        gpu_env_path=args.gpu_env_path,
        recognizer_weights=args.recognizer_weights,
        save_debug_crops_dir=args.save_debug_crops_dir,
        include_full_image_fallback=not args.no_full_image_fallback,
    )


if __name__ == "__main__":
    main()