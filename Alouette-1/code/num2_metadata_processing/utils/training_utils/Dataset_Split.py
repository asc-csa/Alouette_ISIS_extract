from __future__ import annotations

import random
import shutil
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable

import pandas as pd


# CONFIG

LABELS_ROOT = Path(
    r"L:\DATA\ISIS\2026-June-Model-Training\Labels\SSA-23"
)

AUGMENTED_LABELS_CSV = Path(
    r"L:\DATA\ISIS\2026-June-Model-Training\Augmented_Cropped_Images\SSA-23\augmented_metadata_label.csv"
)

PREPROCESSED_IMAGES_ROOT = Path(
    r"L:\DATA\ISIS\2026-June-Model-Training\Preprocessed_Images\SSA-23"
)

OUTPUT_ROOT = Path(
    r"L:\DATA\ISIS\2026-June-Model-Training\Dataset_Split"
)

DATASET_FOLDER_NAME = "SSA-23"

SPLIT_RATIOS: Dict[str, float] = {
    "train": 0.70,
    "validation": 0.20,
    "test": 0.10,
}

RANDOM_SEED = 42
RESET_OUTPUT = False
CONFLICT_POLICY = "exclude"

SUPPORTED_IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp",
}


def validate_configuration() -> None:
    if not LABELS_ROOT.is_dir():
        raise FileNotFoundError(f"Labels folder does not exist: {LABELS_ROOT}")

    if not AUGMENTED_LABELS_CSV.is_file():
        raise FileNotFoundError(
            f"Augmented labels CSV does not exist: {AUGMENTED_LABELS_CSV}"
        )

    if not PREPROCESSED_IMAGES_ROOT.is_dir():
        raise FileNotFoundError(
            f"Preprocessed-images folder does not exist: {PREPROCESSED_IMAGES_ROOT}"
        )

    ratio_total = sum(SPLIT_RATIOS.values())
    if abs(ratio_total - 1.0) > 1e-9:
        raise ValueError(
            f"Split ratios must add up to 1.0, but currently add up to {ratio_total}."
        )

    if any(ratio <= 0 for ratio in SPLIT_RATIOS.values()):
        raise ValueError("Every split ratio must be greater than zero.")

    if CONFLICT_POLICY not in {"exclude", "latest"}:
        raise ValueError("CONFLICT_POLICY must be either 'exclude' or 'latest'.")


def prepare_output_folder() -> None:
    if OUTPUT_ROOT.exists() and any(OUTPUT_ROOT.iterdir()):
        if not RESET_OUTPUT:
            raise FileExistsError(
                f"Output folder is not empty: {OUTPUT_ROOT}\n"
                "To rebuild the split from scratch, set RESET_OUTPUT = True."
            )

        expected_name = "Text_OCR_Dataset_Splits_SSA-23"
        if OUTPUT_ROOT.name != expected_name:
            raise RuntimeError(
                "Refusing to delete an unexpected output folder. "
                f"Expected folder name '{expected_name}', got '{OUTPUT_ROOT.name}'."
            )

        shutil.rmtree(OUTPUT_ROOT)

    for split_name in SPLIT_RATIOS:
        (OUTPUT_ROOT / split_name / DATASET_FOLDER_NAME).mkdir(
            parents=True,
            exist_ok=True,
        )


def discover_label_csv_files() -> list[Path]:
    csv_files = sorted(
        path
        for path in LABELS_ROOT.rglob("*.csv")
        if path.is_file() and not path.name.startswith("_")
    )

    if not csv_files:
        raise FileNotFoundError(
            f"No non-augmented label CSV files were found under: {LABELS_ROOT}"
        )

    return csv_files


def normalize_relative_image_path(value: str) -> str:
    cleaned = str(value).strip().replace("\\", "/")
    relative_path = PurePosixPath(cleaned)

    if not cleaned:
        raise ValueError("relative_image_path is empty")

    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"Unsafe relative_image_path: {value!r}")

    if relative_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported image extension: {value!r}")

    return relative_path.as_posix()


def make_augmented_relative_path(relative_image_path: str) -> str:
    relative_path = PurePosixPath(relative_image_path)

    augmented_filename = (
        f"{relative_path.stem}_augmented"
        f"{relative_path.suffix}"
    )

    return PurePosixPath(
        relative_path.parent,
        augmented_filename,
    ).as_posix()


def read_label_csv(csv_path: Path, is_augmented: bool) -> tuple[pd.DataFrame | None, dict | None]:
    required_columns = {
        "relative_image_path",
        "metadata_label",
        "status",
    }

    try:
        frame = pd.read_csv(
            csv_path,
            dtype=str,
            keep_default_na=False,
            encoding="utf-8-sig",
        )
    except Exception as exc:
        return None, {
            "source_csv": str(csv_path),
            "reason": f"Could not read CSV: {exc}",
        }

    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        return None, {
            "source_csv": str(csv_path),
            "reason": "Missing columns: " + ", ".join(missing_columns),
        }

    frame = frame.copy()
    frame["source_csv"] = str(csv_path)
    frame["source_row_number"] = frame.index + 2
    frame["is_augmented"] = is_augmented

    completed_mask = (
        frame["status"].astype(str).str.strip().str.casefold() == "completed"
    )
    frame = frame.loc[completed_mask].copy()

    if frame.empty:
        return None, None

    frame["metadata_label"] = frame["metadata_label"].astype(str).str.strip()
    frame = frame.loc[frame["metadata_label"] != ""].copy()

    if frame.empty:
        return None, None

    return frame, None


def load_completed_labels(csv_files: Iterable[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    loaded_frames: list[pd.DataFrame] = []
    invalid_csv_records: list[dict[str, str]] = []

    for csv_path in csv_files:
        frame, error_record = read_label_csv(
            csv_path=csv_path,
            is_augmented=False,
        )

        if error_record is not None:
            invalid_csv_records.append(error_record)

        if frame is not None:
            loaded_frames.append(frame)

    augmented_frame, augmented_error = read_label_csv(
        csv_path=AUGMENTED_LABELS_CSV,
        is_augmented=True,
    )

    if augmented_error is not None:
        invalid_csv_records.append(augmented_error)

    if augmented_frame is not None:
        loaded_frames.append(augmented_frame)

    invalid_csv_df = pd.DataFrame(invalid_csv_records)

    if not loaded_frames:
        raise ValueError(
            "No completed, non-empty metadata labels were found."
        )

    combined = pd.concat(loaded_frames, ignore_index=True)
    return combined, invalid_csv_df


def clean_paths_and_find_missing_images(
    labels_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid_rows: list[dict] = []
    excluded_rows: list[dict] = []

    for _, row in labels_df.iterrows():
        record = row.to_dict()

        try:
            normalized_relative_path = normalize_relative_image_path(
                record["relative_image_path"]
            )
        except ValueError as exc:
            record["exclusion_reason"] = str(exc)
            excluded_rows.append(record)
            continue

        is_augmented = bool(record.get("is_augmented", False))

        if is_augmented:
            preprocessed_relative_path = make_augmented_relative_path(
                normalized_relative_path
            )
        else:
            preprocessed_relative_path = normalized_relative_path

        relative_path = PurePosixPath(preprocessed_relative_path)
        source_image_path = PREPROCESSED_IMAGES_ROOT.joinpath(*relative_path.parts)

        if not source_image_path.is_file():
            record["normalized_relative_image_path"] = normalized_relative_path
            record["preprocessed_relative_image_path"] = preprocessed_relative_path
            record["expected_source_image_path"] = str(source_image_path)
            record["exclusion_reason"] = "Preprocessed image was not found"
            excluded_rows.append(record)
            continue

        try:
            file_size = source_image_path.stat().st_size
        except OSError as exc:
            record["normalized_relative_image_path"] = normalized_relative_path
            record["preprocessed_relative_image_path"] = preprocessed_relative_path
            record["expected_source_image_path"] = str(source_image_path)
            record["exclusion_reason"] = f"Could not inspect image file: {exc}"
            excluded_rows.append(record)
            continue

        if file_size == 0:
            record["normalized_relative_image_path"] = normalized_relative_path
            record["preprocessed_relative_image_path"] = preprocessed_relative_path
            record["expected_source_image_path"] = str(source_image_path)
            record["exclusion_reason"] = "Preprocessed image file is empty"
            excluded_rows.append(record)
            continue

        record["relative_image_path"] = normalized_relative_path
        record["preprocessed_relative_image_path"] = preprocessed_relative_path
        record["source_image_path"] = str(source_image_path)
        record["image_group"] = relative_path.parts[0]

        valid_rows.append(record)

    valid_df = pd.DataFrame(valid_rows)
    excluded_df = pd.DataFrame(excluded_rows)

    if valid_df.empty:
        raise ValueError("No valid labeled images remained after path validation.")

    return valid_df, excluded_df


def resolve_duplicate_labels(
    labels_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Keep one row per actual preprocessed image.

    This allows:
        23-008/image086.png
        23-008/image086_augmented.png

    to both exist as separate training examples.
    """
    working = labels_df.copy()

    if "labeled_at" not in working.columns:
        working["labeled_at"] = ""

    working["parsed_labeled_at"] = pd.to_datetime(
        working["labeled_at"],
        errors="coerce",
        utc=True,
    )

    kept_rows: list[pd.Series] = []
    conflict_frames: list[pd.DataFrame] = []

    for _, group in working.groupby("preprocessed_relative_image_path", sort=True):
        unique_labels = sorted(set(group["metadata_label"].astype(str)))

        ordered = group.sort_values(
            by=["parsed_labeled_at", "source_csv", "source_row_number"],
            na_position="first",
        )

        newest_row = ordered.iloc[-1]

        if len(unique_labels) == 1:
            kept_rows.append(newest_row)
            continue

        if CONFLICT_POLICY == "latest":
            kept_rows.append(newest_row)
            continue

        conflict_copy = group.copy()
        conflict_copy["conflicting_labels"] = " | ".join(unique_labels)
        conflict_frames.append(conflict_copy)

    deduplicated = pd.DataFrame(kept_rows).reset_index(drop=True)
    deduplicated = deduplicated.drop(columns=["parsed_labeled_at"], errors="ignore")

    if conflict_frames:
        conflicts = pd.concat(conflict_frames, ignore_index=True)
        conflicts = conflicts.drop(columns=["parsed_labeled_at"], errors="ignore")
    else:
        conflicts = pd.DataFrame()

    if deduplicated.empty:
        raise ValueError("No images remained after resolving duplicate labels.")

    return deduplicated, conflicts



def assign_groups_to_splits(labels_df: pd.DataFrame) -> dict[str, str]:
    """
    Assign an entire film-roll subfolder to exactly one split.

    The greedy objective approximately matches the requested image ratios while
    preventing images from the same subfolder from leaking across splits.
    """
    group_counts = labels_df.groupby("image_group").size().to_dict()

    if len(group_counts) < len(SPLIT_RATIOS):
        raise ValueError(
            "At least three distinct image subfolders are required for grouped "
            f"train/validation/test splitting; found {len(group_counts)}."
        )

    rng = random.Random(RANDOM_SEED)
    group_items = list(group_counts.items())
    rng.shuffle(group_items)
    group_items.sort(key=lambda item: item[1], reverse=True)

    total_images = len(labels_df)
    target_counts = {
        split_name: total_images * ratio
        for split_name, ratio in SPLIT_RATIOS.items()
    }
    current_counts = {split_name: 0 for split_name in SPLIT_RATIOS}
    assignment: dict[str, str] = {}

    for group_name, group_size in group_items:
        candidate_scores: list[tuple[float, float, str]] = []

        for split_name in SPLIT_RATIOS:
            candidate_counts = current_counts.copy()
            candidate_counts[split_name] += group_size

            # Minimize normalized squared distance from all requested targets.
            score = sum(
                (
                    (candidate_counts[name] - target_counts[name])
                    / max(target_counts[name], 1.0)
                )
                ** 2
                for name in SPLIT_RATIOS
            )

            # Secondary key favors the split with the largest remaining deficit.
            remaining_deficit = (
                target_counts[split_name] - current_counts[split_name]
            )
            candidate_scores.append((score, -remaining_deficit, split_name))

        _, _, chosen_split = min(candidate_scores)
        assignment[group_name] = chosen_split
        current_counts[chosen_split] += group_size

    # Repair an empty split for unusual group-size distributions
    # Move the smallest available group from a donor split that contains more than one group. 
    # This preserves the no-leakage rule while guaranteeing that train,val, and test all contain data.
    for empty_split in [
        split_name
        for split_name, count in current_counts.items()
        if count == 0
    ]:
        donor_candidates = []

        for donor_split in SPLIT_RATIOS:
            donor_groups = [
                group_name
                for group_name, assigned_split in assignment.items()
                if assigned_split == donor_split
            ]
            if len(donor_groups) <= 1:
                continue

            normalized_excess = (
                current_counts[donor_split] - target_counts[donor_split]
            ) / max(target_counts[donor_split], 1.0)
            donor_candidates.append(
                (normalized_excess, current_counts[donor_split], donor_split)
            )

        if not donor_candidates:
            raise RuntimeError(
                "Could not populate every split while keeping image subfolders intact."
                "Add more labeled subfolders or revise the split ratios."
            )

        _, _, donor_split = max(donor_candidates)
        donor_groups = [
            group_name
            for group_name, assigned_split in assignment.items()
            if assigned_split == donor_split
        ]
        group_to_move = min(donor_groups, key=lambda name: group_counts[name])
        moved_size = group_counts[group_to_move]

        assignment[group_to_move] = empty_split
        current_counts[donor_split] -= moved_size
        current_counts[empty_split] += moved_size

    return assignment



def copy_images_and_write_manifests(labels_df: pd.DataFrame) -> pd.DataFrame:
    copied_records: list[dict] = []

    for _, row in labels_df.iterrows():
        split_name = row["split"]
        relative_path = PurePosixPath(row["preprocessed_relative_image_path"])
        source_image_path = Path(row["source_image_path"])

        destination_relative_path = PurePosixPath(
            DATASET_FOLDER_NAME,
            *relative_path.parts,
        )
        destination_image_path = (
            OUTPUT_ROOT
            / split_name
            / Path(*destination_relative_path.parts)
        )

        destination_image_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_image_path, destination_image_path)

        output_record = row.to_dict()
        output_record["dataset_relative_image_path"] = (
            destination_relative_path.as_posix()
        )
        output_record["split_image_path"] = str(destination_image_path)
        copied_records.append(output_record)

    manifest = pd.DataFrame(copied_records)

    preferred_columns = [
        "split",
        "dataset_relative_image_path",
        "metadata_label",
        "image_group",
        "relative_image_path",
        "preprocessed_relative_image_path",
        "source_image_path",
        "split_image_path",
        "is_augmented",
        "image_filename",
        "labeler_name",
        "labeled_at",
        "session_id",
        "source_csv",
        "source_row_number",
    ]
    ordered_columns = [
        column for column in preferred_columns if column in manifest.columns
    ]
    remaining_columns = [
        column for column in manifest.columns if column not in ordered_columns
    ]
    manifest = manifest[ordered_columns + remaining_columns]

    for split_name in SPLIT_RATIOS:
        split_manifest = manifest.loc[manifest["split"] == split_name].copy()
        split_manifest = split_manifest.sort_values(
            ["image_group", "dataset_relative_image_path"]
        )
        split_manifest.to_csv(
            OUTPUT_ROOT / split_name / "labels.csv",
            index=False,
            encoding="utf-8-sig",
        )

    manifest = manifest.sort_values(
        ["split", "image_group", "dataset_relative_image_path"]
    ).reset_index(drop=True)
    manifest.to_csv(
        OUTPUT_ROOT / "all_splits.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return manifest



def build_summary(manifest: pd.DataFrame) -> pd.DataFrame:
    total_images = len(manifest)
    summary_rows: list[dict] = []

    for split_name in SPLIT_RATIOS:
        split_df = manifest.loc[manifest["split"] == split_name]
        summary_rows.append(
            {
                "split": split_name,
                "images": len(split_df),
                "image_percentage": round(100 * len(split_df) / total_images, 2),
                "subfolders": split_df["image_group"].nunique(),
                "requested_percentage": 100 * SPLIT_RATIOS[split_name],
            }
        )

    return pd.DataFrame(summary_rows)


# MAIN

def main() -> None:
    validate_configuration()
    prepare_output_folder()

    csv_files = discover_label_csv_files()
    print(f"Found {len(csv_files)} label CSV file(s).")

    raw_labels, invalid_csv_df = load_completed_labels(csv_files)
    print(f"Found {len(raw_labels)} completed label row(s).")

    valid_labels, excluded_images_df = clean_paths_and_find_missing_images(raw_labels)
    deduplicated_labels, conflicts_df = resolve_duplicate_labels(valid_labels)

    group_assignment = assign_groups_to_splits(deduplicated_labels)
    deduplicated_labels["split"] = deduplicated_labels["image_group"].map(
        group_assignment
    )

    manifest = copy_images_and_write_manifests(deduplicated_labels)
    summary = build_summary(manifest)

    summary.to_csv(
        OUTPUT_ROOT / "split_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if not invalid_csv_df.empty:
        invalid_csv_df.to_csv(
            OUTPUT_ROOT / "invalid_csv_files.csv",
            index=False,
            encoding="utf-8-sig",
        )

    if not excluded_images_df.empty:
        excluded_images_df.to_csv(
            OUTPUT_ROOT / "excluded_images.csv",
            index=False,
            encoding="utf-8-sig",
        )

    if not conflicts_df.empty:
        conflicts_df.to_csv(
            OUTPUT_ROOT / "conflicting_labels.csv",
            index=False,
            encoding="utf-8-sig",
        )

    print("\nSplit completed successfully.")
    print(f"Output folder: {OUTPUT_ROOT}")
    print("\nSplit summary:")
    print(summary.to_string(index=False))

    print("\nOutput structure:")
    print(f"  {OUTPUT_ROOT}\\train\\SSA-23\\...")
    print(f"  {OUTPUT_ROOT}\\validation\\SSA-23\\...")
    print(f"  {OUTPUT_ROOT}\\test\\SSA-23\\...")
    print("  Each split folder also contains a labels.csv manifest.")

    if not conflicts_df.empty:
        print(
            f"\nExcluded {conflicts_df['preprocessed_relative_image_path'].nunique()} image(s) "
            "with conflicting labels. See conflicting_labels.csv."
        )

    if not excluded_images_df.empty:
        print(
            f"Excluded {len(excluded_images_df)} row(s) because of missing, "
            "empty, unsupported, or unsafe image paths. See excluded_images.csv."
        )


if __name__ == "__main__":
    main()
