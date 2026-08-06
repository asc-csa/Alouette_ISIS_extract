from pathlib import Path
import random
import csv
import tkinter as tk
from tkinter import simpledialog, messagebox
from PIL import Image, ImageTk, ImageDraw
from datetime import datetime

INPUT_ROOT = Path(r"L:\DATA\ISIS\2026-June-Model-Training\Cropped_Images\SSA-23")
OUTPUT_ROOT = Path(r"L:\DATA\ISIS\2026-June-Model-Training\Augmented_Cropped_Images\SSA-23")
LABELS_ROOT = Path(r"L:\DATA\ISIS\2026-June-Model-Training\Labels\SSA-23")
AUGMENTED_LABEL_CSV = OUTPUT_ROOT / "augmented_metadata_label.csv"
REGISTRY_CSV = Path(r"L:\DATA\ISIS\2026-June-Model-Training\Labels\SSA-23\_image_assignment_registry.csv")

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

EXCLUDED_SUBFOLDERS = {
    "23-004", "23-016", "23-019", "23-022", "23-025",
    "23-044", "23-051", "23-057", "23-062", "23-069", "23-072"
}


def output_exists_for_input(input_path):
    relative = input_path.relative_to(INPUT_ROOT)
    output_path = OUTPUT_ROOT / relative
    return output_path.exists()


def collect_images():
    images = []
    seen = set()

    if not REGISTRY_CSV.exists():
        messagebox.showerror(
            "Registry not found",
            f"Could not find registry file:\n{REGISTRY_CSV}"
        )
        return images

    with open(REGISTRY_CSV, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        for row in reader:
            if row.get("status", "").strip().lower() != "completed":
                continue

            relative_path_str = row.get("relative_image_path", "").strip()
            if not relative_path_str:
                continue

            relative_path = Path(relative_path_str.replace("/", "\\"))
            input_path = INPUT_ROOT / relative_path

            if input_path in seen:
                continue

            if input_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            if any(part in EXCLUDED_SUBFOLDERS for part in input_path.parts):
                continue

            if not input_path.exists():
                continue

            # Do not select images already augmented
            if output_exists_for_input(input_path):
                continue

            images.append(input_path)
            seen.add(input_path)

    return images


def make_output_path(input_path, index=None):
    relative = input_path.relative_to(INPUT_ROOT)
    out_path = OUTPUT_ROOT / relative
    out_path.parent.mkdir(parents=True, exist_ok=True)

    return out_path


def normalize_relative_path(path_str):
    return path_str.strip().replace("\\", "/")


def find_completed_registry_row(input_path):
    target_relative = normalize_relative_path(
        str(input_path.relative_to(INPUT_ROOT))
    )

    with open(REGISTRY_CSV, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        completed_rows = [
            row for row in reader
            if row.get("status", "").strip().lower() == "completed"
            and normalize_relative_path(row.get("relative_image_path", "")) == target_relative
        ]

    if not completed_rows:
        return None

    return completed_rows[-1]


def find_label_file(labeler_name):
    labeler_name_lower = labeler_name.strip().lower()

    for path in LABELS_ROOT.glob("*.csv"):
        if path.name.startswith("_"):
            continue

        if path.name.lower().startswith(labeler_name_lower):
            return path

    return None


def find_original_label_row(input_path):
    registry_row = find_completed_registry_row(input_path)

    if registry_row is None:
        return None

    labeler_name = registry_row.get("labeler_name", "").strip()
    session_id = registry_row.get("session_id", "").strip()
    target_relative = normalize_relative_path(
        str(input_path.relative_to(INPUT_ROOT))
    )

    label_file = find_label_file(labeler_name)

    if label_file is None:
        return None

    with open(label_file, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        for row in reader:
            if row.get("status", "").strip().lower() != "completed":
                continue

            same_path = normalize_relative_path(row.get("relative_image_path", "")) == target_relative
            same_session = row.get("session_id", "").strip() == session_id

            if same_path and same_session:
                return row

    return None


def make_augmented_metadata_label(original_label):
    original_label = str(original_label).strip()

    if len(original_label) < 2:
        return original_label

    last_two_digits = original_label[-2:]
    remaining_digits = original_label[:-2]

    return last_two_digits + remaining_digits


def append_augmented_label_row(input_path, output_path):
    original_row = find_original_label_row(input_path)

    if original_row is None:
        messagebox.showwarning(
            "Label not found",
            f"Could not find completed label data for:\n{input_path}"
        )
        return

    original_label = original_row.get("metadata_label", "").strip()
    augmented_label = make_augmented_metadata_label(original_label)

    output_relative = normalize_relative_path(
        str(output_path.relative_to(OUTPUT_ROOT))
    )

    new_row = {
        "labeler_name": original_row.get("labeler_name", ""),
        "image_filename": output_path.name,
        "relative_image_path": output_relative,
        "full_image_path": str(output_path),
        "metadata_label": augmented_label,
        "status": "completed",
        "labeled_at": datetime.now().isoformat(timespec="seconds"),
        "session_id": original_row.get("session_id", ""),
        "source_relative_image_path": original_row.get("relative_image_path", ""),
        "source_metadata_label": original_label,
    }

    fieldnames = list(new_row.keys())
    write_header = not AUGMENTED_LABEL_CSV.exists()

    with open(AUGMENTED_LABEL_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if write_header:
            writer.writeheader()

        writer.writerow(new_row)


class AugmentationGUI:
    def __init__(self, root, image_paths):
        self.root = root
        self.image_paths = image_paths
        self.index = 0

        self.image_path = None
        self.original = None
        self.tk_image = None
        self.scale = 1.0

        self.step = 1
        self.start_x = None
        self.end_x = None
        self.move_x = None

        self.dragging = False

        self.canvas = tk.Canvas(root, bg="gray")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        controls = tk.Frame(root)
        controls.pack(fill=tk.X)

        tk.Button(controls, text="Reset Current Image", command=self.reset_current).pack(side=tk.LEFT, padx=8, pady=8)
        tk.Button(controls, text="Save", command=self.save).pack(side=tk.LEFT, padx=8, pady=8)
        tk.Button(controls, text="Skip", command=self.next_image).pack(side=tk.LEFT, padx=8, pady=8)

        self.status = tk.Label(controls, text="")
        self.status.pack(side=tk.LEFT, padx=12)

        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

        self.next_image()

    def next_image(self):
        if self.index >= len(self.image_paths):
            messagebox.showinfo("Done", "Augmentation complete.")
            self.root.destroy()
            return

        self.image_path = self.image_paths[self.index]
        self.original = Image.open(self.image_path).convert("RGB")
        self.index += 1

        self.reset_current()

    def reset_current(self):
        self.step = 1
        self.start_x = None
        self.end_x = None
        self.move_x = None
        self.dragging = False
        self.render()

    def render(self):
        preview = self.original.copy()
        draw_original = ImageDraw.Draw(preview)

        if self.start_x is not None:
            draw_original.line([(self.start_x, 0), (self.start_x, preview.height)], fill="white", width=3)

        if self.end_x is not None:
            draw_original.line([(self.end_x, 0), (self.end_x, preview.height)], fill="white", width=3)
            draw_original.rectangle(
                [self.start_x, 0, self.end_x, preview.height],
                outline="white",
                width=2,
            )

        if self.start_x is not None and self.end_x is not None and self.move_x is not None:
            selected = self.original.crop((self.start_x, 0, self.end_x, self.original.height))
            preview.paste(selected, (self.move_x, 0))
            draw_original = ImageDraw.Draw(preview)
            draw_original.rectangle(
                [self.move_x, 0, self.move_x + selected.width, preview.height],
                outline="black",
                width=2,
            )

        max_w = 1350
        max_h = 320
        self.scale = min(max_w / preview.width, max_h / preview.height, 1.0)

        display_w = int(preview.width * self.scale)
        display_h = int(preview.height * self.scale)

        preview = preview.resize((display_w, display_h), Image.Resampling.LANCZOS)

        self.tk_image = ImageTk.PhotoImage(preview)
        self.canvas.delete("all")
        self.canvas.config(width=display_w, height=display_h)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

        if self.step == 1:
            instruction = "Step 1: click the BEGINNING of the last 2 digits"
        elif self.step == 2:
            instruction = "Step 2: click the END of the last 2 digits"
        else:
            instruction = "Step 3: drag the selected digits to right before the original metadata starts, then Save"

        self.status.config(
            text=f"{self.index}/{len(self.image_paths)} | {self.image_path.name} | {instruction}"
        )

    def canvas_x_to_image_x(self, canvas_x):
        x = int(canvas_x / self.scale)
        return max(0, min(x, self.original.width - 1))

    def on_click(self, event):
        x = self.canvas_x_to_image_x(event.x)

        if self.step == 1:
            self.start_x = x
            self.step = 2

        elif self.step == 2:
            self.end_x = x

            if self.end_x <= self.start_x:
                messagebox.showwarning("Invalid selection", "The end must be to the right of the beginning.")
                self.end_x = None
                return

            self.move_x = self.start_x
            self.step = 3

        elif self.step == 3:
            self.move_x = x
            self.dragging = True

        self.render()

    def on_drag(self, event):
        if self.step != 3:
            return

        x = self.canvas_x_to_image_x(event.x)
        selected_width = self.end_x - self.start_x

        self.move_x = max(0, min(x, self.original.width - selected_width))
        self.render()

    def on_release(self, event):
        self.dragging = False

    def make_augmented_image(self):
        selected = self.original.crop(
            (self.start_x, 0, self.end_x, self.original.height)
        )

        # Keep original image only up to right before the original last 2 digits
        new_img = self.original.crop(
            (0, 0, self.start_x, self.original.height)
        ).copy()

        # Paste the selected last 2 digits exactly where the user slid them
        new_img.paste(selected, (self.move_x, 0))

        return new_img

    def save(self):
        if self.start_x is None or self.end_x is None or self.move_x is None:
            messagebox.showwarning("Not complete", "Complete all 3 steps before saving.")
            return

        if self.move_x >= self.start_x:
            messagebox.showwarning(
                "Invalid placement",
                "Move the selected digits to the LEFT of where the last 2 digits originally started."
            )
            return

        output_path = make_output_path(self.image_path, self.index)
        augmented = self.make_augmented_image()
        augmented.save(output_path)
        
        append_augmented_label_row(self.image_path, output_path)

        manifest_path = OUTPUT_ROOT / "augmentation_manifest.csv"
        write_header = not manifest_path.exists()

        with open(manifest_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            if write_header:
                writer.writerow([
                    "source_path",
                    "output_path",
                    "last_two_start_x",
                    "last_two_end_x",
                    "moved_to_x",
                    "original_width",
                    "new_width",
                    "height",
                ])

            writer.writerow([
                str(self.image_path),
                str(output_path),
                self.start_x,
                self.end_x,
                self.move_x,
                self.original.width,
                augmented.width,
                augmented.height,
            ])

        self.next_image()


def main():
    root = tk.Tk()
    root.withdraw()

    count = simpledialog.askinteger(
        "Number of images",
        "How many images do you want to augment?",
        minvalue=1,
    )

    if not count:
        return

    images = collect_images()
    random.shuffle(images)
    images = images[:count]

    if not images:
        messagebox.showerror("Error", "No images found.")
        return

    root.deiconify()
    root.title("Metadata Image Augmentation Tool")

    AugmentationGUI(root, images)
    root.mainloop()


if __name__ == "__main__":
    main()