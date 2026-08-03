"""
Delete_Files_By_Keywords.py

Description:
    Recursively scans a specified root directory and all subdirectories
    for files whose names contain one or more user-supplied keywords.
    Matching files are moved to a designated archive directory instead
    of being permanently deleted.

    The original directory structure is preserved within the archive
    folder to avoid filename collisions and to make recovery easier.

Parameters:
    root_directory (str | pathlib.Path):
        The root directory to scan recursively.

    keywords (list[str]):
        A list of keywords. If any keyword is found within a filename
        (case-insensitive), the file will be archived.

    archive_directory (str | pathlib.Path):
        Location where matching files will be moved.

Example:
    archive_files_by_keywords(
        root_directory="C:/datasets/images",
        keywords=["_augmented", "_backup", "_temp"],
        archive_directory="C:/datasets/archive"
    )

Behavior:
    - Searches all files recursively beneath the root directory.
    - Checks filenames only (not file contents).
    - Preserves the original folder hierarchy in the archive.
    - Creates archive subfolders automatically as needed.
    - Skips files that are already located inside the archive directory.

Disclaimer:
    This script was developed with the assistance of Microsoft Copilot.
    Users are responsible for reviewing, testing, and validating the
    script before running it in a production, business, or data-sensitive
    environment. Always verify the archive location and keyword list
    before execution.
"""

from pathlib import Path
import shutil


def archive_files_by_keywords(root_directory, keywords, archive_directory):
    """
    Recursively archives files whose filenames contain any of the
    specified keywords.

    Args:
        root_directory (str | Path):
            Root directory to scan.

        keywords (list[str]):
            Keywords to search for in filenames.
            Matching is case-insensitive.

        archive_directory (str | Path):
            Destination archive folder.

    Returns:
        int:
            Number of files successfully archived.
    """
    root = Path(root_directory).resolve()
    archive_root = Path(archive_directory).resolve()

    if not root.is_dir():
        raise ValueError(f"'{root}' is not a valid directory.")

    archive_root.mkdir(parents=True, exist_ok=True)

    keywords = [keyword.lower() for keyword in keywords]
    archived_count = 0

    for file_path in root.rglob("*"):
        if not file_path.is_file():
            continue

        # Prevent archived files from being reprocessed
        if archive_root in file_path.parents:
            continue

        filename = file_path.name.lower()

        if any(keyword in filename for keyword in keywords):
            try:
                # Preserve original directory structure
                relative_path = file_path.relative_to(root)
                destination = archive_root / relative_path

                destination.parent.mkdir(parents=True, exist_ok=True)

                shutil.move(str(file_path), str(destination))

                print(f"Archived: {file_path} -> {destination}")
                archived_count += 1

            except Exception as e:
                print(f"Failed to archive {file_path}: {e}")

    print(f"\nDone. Archived {archived_count} file(s).")
    return archived_count


###################################################################
# MAIN
###################################################################
ROOT_DIRECTORY = Path(
    r"L:\DATA\ISIS\Imageupload_20260428_Processing\03_Text_Detection\labels"
)
KEYWORRDS = ['_augmented']

ARCHIVE_DIRECTORY = Path(
    r"L:\DATA\ISIS\Imageupload_20260428_Processing\Removed_Dataset_Augmentation\Yolo_Labels"
)

if __name__ == "__main__":
    archive_files_by_keywords(
        root_directory=ROOT_DIRECTORY,
        keywords=KEYWORRDS,
        archive_directory=ARCHIVE_DIRECTORY
    )