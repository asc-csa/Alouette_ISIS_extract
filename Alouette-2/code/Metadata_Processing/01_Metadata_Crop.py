from pathlib import Path
import cv2

INPUT_ROOT = Path(
    r"L:\DATA\ISIS\Imageupload_20260428\CSA_2026-04-07\2\SSA-23"
)

OUTPUT_ROOT = Path(
    r"L:\DATA\ISIS\2026-June-Model-Training\Cropped_Images\SSA-23"
)

# Number of pixel rows to keep from the bottom of each image - obtained from EDA analysis in 00_EDA.ipynb
CROP_HEIGHT = 100

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
}


def crop_bottom_recursively(
    input_root: Path,
    output_root: Path,
    crop_height: int,
) -> None:
    if not input_root.exists():
        raise FileNotFoundError(
            f"Input directory does not exist: {input_root}"
        )

    if crop_height <= 0:
        raise ValueError("Crop height must be greater than zero.")

    output_root.mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped = 0
    failed = 0

    for image_path in input_root.rglob("*"):
        if not image_path.is_file():
            continue

        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        # Path of the image relative to the input root
        relative_path = image_path.relative_to(input_root)

        # Recreate the same folder structure under the output root
        output_path = output_root / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.exists() and output_path.stat().st_size > 0:
            print(f"[SKIPPED: ALREADY EXISTS] {output_path}")
            skipped += 1
            continue

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_UNCHANGED,
        )

        if image is None:
            print(f"[FAILED TO READ] {image_path}")
            failed += 1
            continue

        image_height = image.shape[0]

        if image_height < crop_height:
            print(
                f"[SKIPPED: TOO SHORT] {image_path} has "
                f"{image_height} rows; at least {crop_height} are required."
            )
            skipped += 1
            continue

        # Keep every column and the bottom 80 pixel rows
        cropped_image = image[-crop_height:, ...]

        saved = cv2.imwrite(
            str(output_path),
            cropped_image,
        )

        if not saved:
            print(f"[FAILED TO SAVE] {output_path}")
            failed += 1
            continue

        processed += 1

        if processed % 100 == 0:
            print(f"Processed {processed} images...")

    print("\nFinished")
    print(f"Successfully cropped: {processed}")
    print(f"Skipped:              {skipped}")
    print(f"Failed:               {failed}")
    print(f"Output directory:     {output_root}")


if __name__ == "__main__":
    crop_bottom_recursively(
        input_root=INPUT_ROOT,
        output_root=OUTPUT_ROOT,
        crop_height=CROP_HEIGHT,
    )
