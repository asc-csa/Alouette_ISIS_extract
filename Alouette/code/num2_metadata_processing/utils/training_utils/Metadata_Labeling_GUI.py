"""
GUI tool for manually labeling metadata in cropped ISIS images.
This code was generated with the help of AI tools.

Features
--------
- Prompts for the labeler's name and desired number of images.
- Randomly selects images across all nested subfolders.
- Prevents active image assignments from overlapping across users.
- Never assigns the same image to the same user twice.
- Appends completed, skipped, abandoned, and load-error records to a
  per-user CSV file.
- Saves each action immediately.
- Supports zoom controls, mouse-wheel scrolling, and Ctrl+mouse-wheel zoom.

Dependency
----------
pip install pillow
conda install pillow
"""

from __future__ import annotations

import csv
import os
import random
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, Optional, Set, Tuple

import tkinter as tk
from tkinter import messagebox, ttk

try:
    from PIL import Image, ImageTk
except ImportError as exc:
    raise SystemExit(
        "Pillow is required. Install it with: pip install pillow OR conda install pillow"
    ) from exc


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

IMAGE_ROOT = Path(
    r"L:\DATA\ISIS\2026-June-Model-Training\Cropped_Images\SSA-23"
)

LABEL_ROOT = Path(
    r"L:\DATA\ISIS\2026-June-Model-Training\Labels\SSA-23"
)

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
}

EXCLUDED_SUBFOLDERS = {
    "23-004",
    "23-016",
    "23-019",
    "23-022",
    "23-025",
    "23-044",
    "23-051",
    "23-057",
    "23-062",
    "23-069",
    "23-072",
}

REGISTRY_PATH = LABEL_ROOT / "_image_assignment_registry.csv"
LOCK_PATH = LABEL_ROOT / "_image_assignment_registry.lock"

# A reservation older than this is treated as abandoned, which allows another
# user to receive the image after a computer crash or forced shutdown.
RESERVATION_TIMEOUT = timedelta(hours=24)

# Lock operations are normally very short. A lock file older than this is
# assumed to have been left behind by a crashed process.
STALE_LOCK_SECONDS = 120
LOCK_WAIT_SECONDS = 20

USER_CSV_FIELDS = [
    "labeler_name",
    "image_filename",
    "relative_image_path",
    "full_image_path",
    "metadata_label",
    "status",
    "labeled_at",
    "session_id",
]

REGISTRY_FIELDS = [
    "relative_image_path",
    "image_filename",
    "labeler_name",
    "session_id",
    "status",
    "timestamp",
]

GLOBAL_FINAL_STATUSES = {
    "completed",
    "skipped",
    "load_error",
}

USER_SEEN_STATUSES = {
    "completed",
    "skipped",
    "abandoned",
    "load_error",
}

INVALID_WINDOWS_FILENAME_CHARS = set('<>:"/\\|?*')


# ---------------------------------------------------------------------------
# Shared-file helpers
# ---------------------------------------------------------------------------

class SharedFileLock:
    """
    Simple lock based on atomic lock-file creation.

    This is suitable for a shared Windows/network directory as long as all
    labelers use this script and point to the same LABEL_ROOT.
    """

    def __init__(
        self,
        lock_path: Path,
        wait_seconds: int = LOCK_WAIT_SECONDS,
        stale_seconds: int = STALE_LOCK_SECONDS,
    ) -> None:
        self.lock_path = lock_path
        self.wait_seconds = wait_seconds
        self.stale_seconds = stale_seconds
        self._acquired = False

    def __enter__(self) -> "SharedFileLock":
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + self.wait_seconds

        while True:
            try:
                file_descriptor = os.open(
                    str(self.lock_path),
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                )
                with os.fdopen(file_descriptor, "w", encoding="utf-8") as file:
                    file.write(
                        f"pid={os.getpid()}\n"
                        f"created={datetime.now().astimezone().isoformat()}\n"
                    )
                    file.flush()
                    os.fsync(file.fileno())

                self._acquired = True
                return self

            except FileExistsError:
                try:
                    age_seconds = (
                        time.time() - self.lock_path.stat().st_mtime
                    )
                    if age_seconds > self.stale_seconds:
                        self.lock_path.unlink(missing_ok=True)
                        continue
                except FileNotFoundError:
                    continue
                except OSError:
                    pass

                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "Could not obtain the shared labeling lock. "
                        "Another labeling session may be updating the files."
                    )

                time.sleep(0.1)

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._acquired:
            try:
                self.lock_path.unlink(missing_ok=True)
            finally:
                self._acquired = False


def append_csv_row(
    csv_path: Path,
    fieldnames: Iterable[str],
    row: Dict[str, str],
) -> None:
    """Append one row and create the header when the file is new."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(fieldnames)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0

    with csv_path.open(
        "a",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )

        if write_header:
            writer.writeheader()

        writer.writerow(row)
        file.flush()
        os.fsync(file.fileno())


def read_csv_rows(csv_path: Path) -> Iterable[Dict[str, str]]:
    """Read CSV rows safely. A malformed or temporarily empty file is skipped."""
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return []

    try:
        with csv_path.open(
            "r",
            newline="",
            encoding="utf-8-sig",
        ) as file:
            return list(csv.DictReader(file))
    except (OSError, csv.Error, UnicodeError):
        return []


def normalize_relative_path(path: Path) -> str:
    """Store relative paths with forward slashes for consistent matching."""
    return path.as_posix()


def now_text() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def parse_timestamp(value: str) -> Optional[datetime]:
    try:
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            parsed = parsed.astimezone()
        return parsed
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

class MetadataLabelerApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("ISIS Metadata Labeling Tool")
        self.root.geometry("1250x850")
        self.root.minsize(900, 650)
        self.root.protocol("WM_DELETE_WINDOW", self.on_window_close)

        self.labeler_name = ""
        self.target_count = 0
        self.session_id = ""
        self.user_csv_path: Optional[Path] = None

        self.image_lookup: Dict[str, Path] = {}
        self.current_relative_path: Optional[str] = None
        self.current_full_path: Optional[Path] = None

        self.original_image: Optional[Image.Image] = None
        self.display_photo: Optional[ImageTk.PhotoImage] = None
        self.zoom_factor = 1.0

        self.processed_count = 0
        self.completed_count = 0
        self.skipped_count = 0
        self.load_error_count = 0
        self.session_active = False
        self.is_finishing = False

        self.start_frame: Optional[ttk.Frame] = None
        self.label_frame: Optional[ttk.Frame] = None

        self.name_var = tk.StringVar()
        self.count_var = tk.StringVar()
        self.metadata_var = tk.StringVar()
        self.progress_var = tk.StringVar()
        self.path_var = tk.StringVar()
        self.summary_var = tk.StringVar()
        self.zoom_var = tk.StringVar(value="100%")

        self.canvas: Optional[tk.Canvas] = None
        self.metadata_entry: Optional[ttk.Entry] = None

        self.show_start_screen()

    # ------------------------------------------------------------------
    # Startup screen
    # ------------------------------------------------------------------

    def clear_root(self) -> None:
        for child in self.root.winfo_children():
            child.destroy()

    def show_start_screen(self) -> None:
        self.clear_root()
        self.root.unbind("<Return>")

        self.start_frame = ttk.Frame(self.root, padding=24)
        self.start_frame.pack(fill="both", expand=True)

        ttk.Label(
            self.start_frame,
            text="ISIS Metadata Labeling Tool",
            font=("Segoe UI", 20, "bold"),
        ).pack(anchor="w", pady=(0, 18))

        instructions = (
            "Instructions\n\n"
            "• Enter your name and the number of images you want to review "
            "during this session, e.g. bob.\n"
            "• For each image, enter the complete metadata number sequence "
            "that you can read.\n"
            "• The metadata field accepts integer digits only.\n"
            "• If the image is unclear or you cannot read the entire sequence, "
            "click Skip.\n"
            "• Click Complete Session whenever you want to stop.\n"
            "• Every completed, skipped, or abandoned action is saved "
            "immediately.\n"
            "• Images are assigned randomly across all subfolders and are "
            "protected from overlapping active assignments."
        )

        instruction_box = ttk.LabelFrame(
            self.start_frame,
            text="Before you begin",
            padding=16,
        )
        instruction_box.pack(fill="x", pady=(0, 20))

        ttk.Label(
            instruction_box,
            text=instructions,
            justify="left",
            wraplength=1000,
        ).pack(anchor="w")

        form = ttk.Frame(self.start_frame)
        form.pack(anchor="w", fill="x")

        ttk.Label(
            form,
            text="Your name:",
        ).grid(row=0, column=0, sticky="w", padx=(0, 12), pady=8)

        name_entry = ttk.Entry(
            form,
            textvariable=self.name_var,
            width=40,
        )
        name_entry.grid(row=0, column=1, sticky="w", pady=8)

        ttk.Label(
            form,
            text="Number of images:",
        ).grid(row=1, column=0, sticky="w", padx=(0, 12), pady=8)

        count_validation = (
            self.root.register(self.validate_digits_or_empty),
            "%P",
        )

        count_entry = ttk.Entry(
            form,
            textvariable=self.count_var,
            width=15,
            validate="key",
            validatecommand=count_validation,
        )
        count_entry.grid(row=1, column=1, sticky="w", pady=8)

        ttk.Button(
            self.start_frame,
            text="Start Labeling",
            command=self.start_session,
        ).pack(anchor="w", pady=(20, 0), ipadx=14, ipady=6)

        self.root.bind("<Return>", lambda event: self.start_session())
        name_entry.focus_set()

    @staticmethod
    def validate_digits_or_empty(proposed_value: str) -> bool:
        return proposed_value == "" or proposed_value.isdigit()

    @staticmethod
    def validate_labeler_name(name: str) -> Optional[str]:
        if not name:
            return "Please enter your name."

        if name in {".", ".."}:
            return "Please enter a valid name."

        if any(char in INVALID_WINDOWS_FILENAME_CHARS for char in name):
            return (
                'The name cannot contain any of these filename characters: '
                '< > : " / \\ | ? *'
            )

        if name.endswith(" ") or name.endswith("."):
            return "The name cannot end with a space or period."

        return None

    def discover_images(self) -> Dict[str, Path]:
        if not IMAGE_ROOT.exists():
            raise FileNotFoundError(
                f"Image folder does not exist:\n{IMAGE_ROOT}"
            )

        image_lookup: Dict[str, Path] = {}

        for path in IMAGE_ROOT.rglob("*"):
            if not path.is_file():
                continue

            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            relative_path = path.relative_to(IMAGE_ROOT)

            if any(
                folder_name in EXCLUDED_SUBFOLDERS
                for folder_name in relative_path.parts[:-1]
            ):
                continue

            relative = normalize_relative_path(relative_path)
            image_lookup[relative] = path

        return image_lookup

    def start_session(self) -> None:
        name = self.name_var.get().strip()
        count_text = self.count_var.get().strip()

        name_error = self.validate_labeler_name(name)
        if name_error:
            messagebox.showerror("Invalid name", name_error)
            return

        if not count_text:
            messagebox.showerror(
                "Invalid image count",
                "Please enter the number of images for this session.",
            )
            return

        try:
            target_count = int(count_text)
        except ValueError:
            messagebox.showerror(
                "Invalid image count",
                "The number of images must be a positive integer.",
            )
            return

        if target_count <= 0:
            messagebox.showerror(
                "Invalid image count",
                "The number of images must be greater than zero.",
            )
            return

        try:
            LABEL_ROOT.mkdir(parents=True, exist_ok=True)
            self.image_lookup = self.discover_images()
        except (OSError, FileNotFoundError) as exc:
            messagebox.showerror("Cannot start session", str(exc))
            return

        if not self.image_lookup:
            messagebox.showerror(
                "No images found",
                f"No supported image files were found under:\n{IMAGE_ROOT}",
            )
            return

        self.labeler_name = name
        self.target_count = target_count
        self.session_id = uuid.uuid4().hex
        self.user_csv_path = LABEL_ROOT / f"{name}_metadata_label.csv"

        self.processed_count = 0
        self.completed_count = 0
        self.skipped_count = 0
        self.load_error_count = 0
        self.session_active = True
        self.is_finishing = False

        self.build_labeling_screen()
        self.load_next_image()

    # ------------------------------------------------------------------
    # Assignment and persistence
    # ------------------------------------------------------------------

    def read_global_state(
        self,
    ) -> Tuple[Set[str], Set[str], Set[str]]:
        """
        Return:
        - globally finalized image paths,
        - actively reserved image paths,
        - all image paths ever assigned to the current user.
        """
        finalized: Set[str] = set()
        active_reservations: Set[str] = set()
        ever_assigned_to_current_user: Set[str] = set()

        # Read all per-user CSVs so older labels remain respected even if they
        # were created before the registry file existed.
        for csv_path in LABEL_ROOT.glob("*_metadata_label.csv"):
            for row in read_csv_rows(csv_path):
                relative = (row.get("relative_image_path") or "").strip()
                status = (row.get("status") or "").strip().lower()
                labeler = (row.get("labeler_name") or "").strip()

                if not relative:
                    continue

                if status in GLOBAL_FINAL_STATUSES:
                    finalized.add(relative)

                if (
                    labeler.casefold() == self.labeler_name.casefold()
                    and status in USER_SEEN_STATUSES
                ):
                    ever_assigned_to_current_user.add(relative)

        latest_registry_row: Dict[str, Dict[str, str]] = {}

        for row in read_csv_rows(REGISTRY_PATH):
            relative = (row.get("relative_image_path") or "").strip()
            if not relative:
                continue

            latest_registry_row[relative] = row

            labeler = (row.get("labeler_name") or "").strip()
            status = (row.get("status") or "").strip().lower()

            # Any recorded reservation means this user has already been shown
            # this image, even if the program later crashed.
            if (
                labeler.casefold() == self.labeler_name.casefold()
                and status == "reserved"
            ):
                ever_assigned_to_current_user.add(relative)

        current_time = datetime.now().astimezone()

        for relative, row in latest_registry_row.items():
            status = (row.get("status") or "").strip().lower()

            if status in GLOBAL_FINAL_STATUSES:
                finalized.add(relative)
                continue

            if status != "reserved":
                continue

            reserved_at = parse_timestamp(row.get("timestamp", ""))

            # A missing timestamp is treated as active to avoid accidental
            # duplicate assignments.
            if reserved_at is None:
                active_reservations.add(relative)
                continue

            if current_time - reserved_at <= RESERVATION_TIMEOUT:
                active_reservations.add(relative)

        return finalized, active_reservations, ever_assigned_to_current_user

    def reserve_random_image(self) -> Optional[Tuple[str, Path]]:
        if self.user_csv_path is None:
            raise RuntimeError("The user CSV path has not been initialized.")

        with SharedFileLock(LOCK_PATH):
            (
                finalized,
                active_reservations,
                user_seen,
            ) = self.read_global_state()

            candidates = [
                relative
                for relative in self.image_lookup
                if relative not in finalized
                and relative not in active_reservations
                and relative not in user_seen
            ]

            if not candidates:
                return None

            relative = random.choice(candidates)
            full_path = self.image_lookup[relative]
            timestamp = now_text()

            append_csv_row(
                REGISTRY_PATH,
                REGISTRY_FIELDS,
                {
                    "relative_image_path": relative,
                    "image_filename": full_path.name,
                    "labeler_name": self.labeler_name,
                    "session_id": self.session_id,
                    "status": "reserved",
                    "timestamp": timestamp,
                },
            )

            return relative, full_path

    def save_current_action(
        self,
        status: str,
        metadata_label: str = "",
        count_toward_target: bool = True,
    ) -> None:
        if (
            self.current_relative_path is None
            or self.current_full_path is None
            or self.user_csv_path is None
        ):
            return

        timestamp = now_text()
        relative = self.current_relative_path
        full_path = self.current_full_path

        user_row = {
            "labeler_name": self.labeler_name,
            "image_filename": full_path.name,
            "relative_image_path": relative,
            "full_image_path": str(full_path),
            "metadata_label": metadata_label,
            "status": status,
            "labeled_at": timestamp,
            "session_id": self.session_id,
        }

        registry_row = {
            "relative_image_path": relative,
            "image_filename": full_path.name,
            "labeler_name": self.labeler_name,
            "session_id": self.session_id,
            "status": status,
            "timestamp": timestamp,
        }

        with SharedFileLock(LOCK_PATH):
            append_csv_row(
                self.user_csv_path,
                USER_CSV_FIELDS,
                user_row,
            )
            append_csv_row(
                REGISTRY_PATH,
                REGISTRY_FIELDS,
                registry_row,
            )

        if count_toward_target:
            self.processed_count += 1

        if status == "completed":
            self.completed_count += 1
        elif status == "skipped":
            self.skipped_count += 1
        elif status == "load_error":
            self.load_error_count += 1

        self.current_relative_path = None
        self.current_full_path = None
        self.original_image = None
        self.display_photo = None
        self.metadata_var.set("")

    def abandon_current_image(self) -> None:
        """
        Record that this user saw the image but did not submit a label.

        The registry's latest status becomes "abandoned", so another user may
        receive the image. The current user will not receive it again because
        their CSV and reservation history record that they already saw it.
        """
        if self.current_relative_path is None:
            return

        try:
            self.save_current_action(
                status="abandoned",
                metadata_label="",
                count_toward_target=False,
            )
        except (OSError, TimeoutError) as exc:
            messagebox.showerror(
                "Could not save progress",
                f"The current image could not be marked as abandoned:\n{exc}",
            )

    # ------------------------------------------------------------------
    # Labeling screen
    # ------------------------------------------------------------------

    def build_labeling_screen(self) -> None:
        self.clear_root()
        self.root.unbind("<Return>")

        self.label_frame = ttk.Frame(self.root, padding=10)
        self.label_frame.pack(fill="both", expand=True)

        top_bar = ttk.Frame(self.label_frame)
        top_bar.pack(fill="x", pady=(0, 8))

        ttk.Label(
            top_bar,
            textvariable=self.progress_var,
            font=("Segoe UI", 12, "bold"),
        ).pack(side="left")

        ttk.Label(
            top_bar,
            textvariable=self.summary_var,
        ).pack(side="right")

        path_bar = ttk.Frame(self.label_frame)
        path_bar.pack(fill="x", pady=(0, 8))

        ttk.Label(
            path_bar,
            text="Image:",
            font=("Segoe UI", 9, "bold"),
        ).pack(side="left")

        ttk.Label(
            path_bar,
            textvariable=self.path_var,
        ).pack(side="left", padx=(6, 0))

        viewer_frame = ttk.Frame(self.label_frame)
        viewer_frame.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(
            viewer_frame,
            background="#202020",
            highlightthickness=0,
        )

        vertical_scrollbar = ttk.Scrollbar(
            viewer_frame,
            orient="vertical",
            command=self.canvas.yview,
        )

        horizontal_scrollbar = ttk.Scrollbar(
            viewer_frame,
            orient="horizontal",
            command=self.canvas.xview,
        )

        self.canvas.configure(
            yscrollcommand=vertical_scrollbar.set,
            xscrollcommand=horizontal_scrollbar.set,
        )

        viewer_frame.rowconfigure(0, weight=1)
        viewer_frame.columnconfigure(0, weight=1)

        self.canvas.grid(row=0, column=0, sticky="nsew")
        vertical_scrollbar.grid(row=0, column=1, sticky="ns")
        horizontal_scrollbar.grid(row=1, column=0, sticky="ew")

        controls = ttk.Frame(self.label_frame)
        controls.pack(fill="x", pady=(10, 4))

        ttk.Button(
            controls,
            text="Zoom Out",
            command=self.zoom_out,
        ).pack(side="left")

        ttk.Button(
            controls,
            text="Zoom In",
            command=self.zoom_in,
        ).pack(side="left", padx=(6, 0))

        ttk.Button(
            controls,
            text="Fit to Window",
            command=self.fit_to_window,
        ).pack(side="left", padx=(6, 0))

        ttk.Button(
            controls,
            text="100%",
            command=self.reset_zoom,
        ).pack(side="left", padx=(6, 0))

        ttk.Label(
            controls,
            textvariable=self.zoom_var,
        ).pack(side="left", padx=(10, 0))

        input_frame = ttk.LabelFrame(
            self.label_frame,
            text="Metadata number",
            padding=10,
        )
        input_frame.pack(fill="x", pady=(6, 0))

        metadata_validation = (
            self.root.register(self.validate_digits_or_empty),
            "%P",
        )

        self.metadata_entry = ttk.Entry(
            input_frame,
            textvariable=self.metadata_var,
            validate="key",
            validatecommand=metadata_validation,
            font=("Consolas", 16),
        )
        self.metadata_entry.pack(
            side="left",
            fill="x",
            expand=True,
            padx=(0, 10),
        )

        ttk.Button(
            input_frame,
            text="Submit / Enter",
            command=self.submit_metadata,
        ).pack(side="left", ipadx=8)

        ttk.Button(
            input_frame,
            text="Skip",
            command=self.skip_image,
        ).pack(side="left", padx=(8, 0), ipadx=8)

        ttk.Button(
            input_frame,
            text="Complete Session",
            command=self.complete_session,
        ).pack(side="left", padx=(8, 0), ipadx=8)

        self.metadata_entry.bind(
            "<Return>",
            lambda event: self.submit_metadata(),
        )

        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)
        self.canvas.bind(
            "<Control-MouseWheel>",
            self.on_ctrl_mouse_wheel,
        )

    def update_status_labels(self) -> None:
        next_number = min(
            self.processed_count + 1,
            self.target_count,
        )

        self.progress_var.set(
            f"Image {next_number} of {self.target_count}"
        )

        self.summary_var.set(
            f"Completed: {self.completed_count}   "
            f"Skipped: {self.skipped_count}   "
            f"Load errors: {self.load_error_count}"
        )

        self.path_var.set(self.current_relative_path or "")

    def load_next_image(self) -> None:
        if not self.session_active or self.is_finishing:
            return

        if self.processed_count >= self.target_count:
            self.finish_session(
                reason="You reached your requested number of images."
            )
            return

        try:
            reserved = self.reserve_random_image()
        except (OSError, TimeoutError) as exc:
            messagebox.showerror(
                "Could not reserve an image",
                str(exc),
            )
            self.finish_session(
                reason="The session stopped because an image could not be reserved."
            )
            return

        if reserved is None:
            self.finish_session(
                reason=(
                    "No eligible images remain. All images are either already "
                    "processed, actively assigned, or previously shown to you."
                )
            )
            return

        self.current_relative_path, self.current_full_path = reserved
        self.update_status_labels()

        try:
            with Image.open(self.current_full_path) as image:
                self.original_image = image.convert("RGB").copy()
        except (OSError, ValueError) as exc:
            try:
                self.save_current_action(
                    status="load_error",
                    metadata_label="",
                    count_toward_target=False,
                )
            except (OSError, TimeoutError) as save_exc:
                messagebox.showerror(
                    "Image and save error",
                    f"Could not open the image:\n{exc}\n\n"
                    f"Could not record the load error:\n{save_exc}",
                )
                self.finish_session(
                    reason="The session stopped after an image load error."
                )
                return

            self.root.after(10, self.load_next_image)
            return

        self.zoom_factor = 1.0
        self.render_image()
        self.root.after(100, self.fit_to_window)

        self.metadata_var.set("")
        if self.metadata_entry is not None:
            self.metadata_entry.focus_set()

    # ------------------------------------------------------------------
    # Image viewer
    # ------------------------------------------------------------------

    def render_image(self) -> None:
        if self.canvas is None or self.original_image is None:
            return

        width = max(
            1,
            int(self.original_image.width * self.zoom_factor),
        )
        height = max(
            1,
            int(self.original_image.height * self.zoom_factor),
        )

        resized = self.original_image.resize(
            (width, height),
            Image.Resampling.LANCZOS,
        )

        self.display_photo = ImageTk.PhotoImage(resized)

        self.canvas.delete("all")
        self.canvas.create_image(
            0,
            0,
            anchor="nw",
            image=self.display_photo,
        )
        self.canvas.configure(scrollregion=(0, 0, width, height))
        self.zoom_var.set(f"{self.zoom_factor * 100:.0f}%")

    def set_zoom(self, new_zoom: float) -> None:
        self.zoom_factor = max(0.05, min(new_zoom, 10.0))
        self.render_image()

    def zoom_in(self) -> None:
        self.set_zoom(self.zoom_factor * 1.25)

    def zoom_out(self) -> None:
        self.set_zoom(self.zoom_factor / 1.25)

    def reset_zoom(self) -> None:
        self.set_zoom(1.0)

    def fit_to_window(self) -> None:
        if self.canvas is None or self.original_image is None:
            return

        self.canvas.update_idletasks()

        canvas_width = max(self.canvas.winfo_width() - 4, 1)
        canvas_height = max(self.canvas.winfo_height() - 4, 1)

        width_scale = canvas_width / self.original_image.width
        height_scale = canvas_height / self.original_image.height

        self.set_zoom(min(width_scale, height_scale))

    def on_ctrl_mouse_wheel(self, event: tk.Event) -> str:
        if event.delta > 0:
            self.zoom_in()
        elif event.delta < 0:
            self.zoom_out()

        return "break"

    def on_mouse_wheel(self, event: tk.Event) -> str:
        if self.canvas is not None:
            self.canvas.yview_scroll(
                int(-1 * (event.delta / 120)),
                "units",
            )
        return "break"

    # ------------------------------------------------------------------
    # User actions
    # ------------------------------------------------------------------

    def submit_metadata(self) -> None:
        if not self.session_active or self.current_relative_path is None:
            return

        metadata = self.metadata_var.get().strip()

        if not metadata:
            messagebox.showwarning(
                "Metadata required",
                "Enter the complete integer sequence, or click Skip.",
            )
            return

        if not metadata.isdigit():
            messagebox.showwarning(
                "Digits only",
                "The metadata field accepts integer digits only.",
            )
            return

        try:
            self.save_current_action(
                status="completed",
                metadata_label=metadata,
                count_toward_target=True,
            )
        except (OSError, TimeoutError) as exc:
            messagebox.showerror(
                "Could not save label",
                f"The label was not saved:\n{exc}",
            )
            return

        self.update_status_labels()
        self.load_next_image()

    def skip_image(self) -> None:
        if not self.session_active or self.current_relative_path is None:
            return

        try:
            self.save_current_action(
                status="skipped",
                metadata_label="",
                count_toward_target=True,
            )
        except (OSError, TimeoutError) as exc:
            messagebox.showerror(
                "Could not save skip",
                f"The skipped image was not saved:\n{exc}",
            )
            return

        self.update_status_labels()
        self.load_next_image()

    def complete_session(self) -> None:
        if not self.session_active or self.is_finishing:
            return

        self.finish_session(
            reason="You ended the session early."
        )

    def finish_session(self, reason: str) -> None:
        if self.is_finishing:
            return

        self.is_finishing = True

        if self.current_relative_path is not None:
            self.abandon_current_image()

        self.session_active = False

        output_path = (
            str(self.user_csv_path)
            if self.user_csv_path is not None
            else str(LABEL_ROOT)
        )

        message = (
            f"{reason}\n\n"
            f"Completed: {self.completed_count}\n"
            f"Skipped: {self.skipped_count}\n"
            f"Load errors: {self.load_error_count}\n\n"
            f"CSV file:\n{output_path}"
        )

        messagebox.showinfo("Session complete", message)
        self.show_finished_screen(message)

    def show_finished_screen(self, message: str) -> None:
        self.clear_root()
        self.root.unbind("<Return>")

        frame = ttk.Frame(self.root, padding=30)
        frame.pack(fill="both", expand=True)

        ttk.Label(
            frame,
            text="Labeling session complete",
            font=("Segoe UI", 20, "bold"),
        ).pack(anchor="w", pady=(0, 18))

        ttk.Label(
            frame,
            text=message,
            justify="left",
            wraplength=1000,
        ).pack(anchor="w")

        button_bar = ttk.Frame(frame)
        button_bar.pack(anchor="w", pady=(24, 0))

        ttk.Button(
            button_bar,
            text="Start Another Session",
            command=self.reset_for_new_session,
        ).pack(side="left", ipadx=8, ipady=4)

        ttk.Button(
            button_bar,
            text="Exit",
            command=self.root.destroy,
        ).pack(side="left", padx=(10, 0), ipadx=8, ipady=4)

    def reset_for_new_session(self) -> None:
        self.current_relative_path = None
        self.current_full_path = None
        self.original_image = None
        self.display_photo = None
        self.metadata_var.set("")
        self.count_var.set("")
        self.is_finishing = False
        self.session_active = False
        self.show_start_screen()

    def on_window_close(self) -> None:
        if self.session_active and not self.is_finishing:
            self.is_finishing = True
            if self.current_relative_path is not None:
                self.abandon_current_image()
            self.session_active = False

        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    MetadataLabelerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
