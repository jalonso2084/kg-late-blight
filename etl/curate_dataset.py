import os
import glob
import random
import shutil
import csv

# --- IMPORTANT: UPDATE THIS PATH ---
# This should be the path to the folder where you extracted the 'healthy' and 'lateblt' folders.
SOURCE_PATH = "C:/Users/jorge/OneDrive/Desktop/temp_images"
# -----------------------------------

# Define project paths
PROJECT_ROOT = os.path.abspath('.')
DATA_PATH = os.path.join(PROJECT_ROOT, 'data')
IMAGES_PATH = os.path.join(DATA_PATH, 'images')
LABELS_CSV_PATH = os.path.join(DATA_PATH, 'labels.csv')

# Define split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
# TEST_RATIO is the remainder (0.15)

def process_images(class_name: str, writer):
    """Finds, splits, copies, and logs images for a given class."""
    source_folder = os.path.join(SOURCE_PATH, class_name)

    if not os.path.isdir(source_folder):
        print(f"ERROR: Source folder not found at '{source_folder}'")
        return 0

    all_images = glob.glob(os.path.join(source_folder, '*.jpg'))
    random.shuffle(all_images)

    total_images = len(all_images)
    if total_images == 0:
        print(f"No .jpg images found in '{source_folder}'.")
        return 0

    print(f"Found {total_images} images for class '{class_name}'.")

    train_end = int(total_images * TRAIN_RATIO)
    val_end = train_end + int(total_images * VAL_RATIO)

    splits = {
        'train': all_images[:train_end],
        'val': all_images[train_end:val_end],
        'test': all_images[val_end:]
    }

    for split_name, files in splits.items():
        dest_folder = os.path.join(IMAGES_PATH, split_name)
        print(f"  Copying {len(files)} files to '{split_name}' folder...")
        for file_path in files:
            filename = os.path.basename(file_path)
            new_path = os.path.join(dest_folder, filename)

            # --- THIS IS THE CHANGE: Use copy2 instead of move ---
            shutil.copy2(file_path, new_path)

            relative_path = os.path.join('data/images', split_name, filename).replace('\\', '/')
            writer.writerow([relative_path, class_name, 'tanzania_field_1', '2024-01-01'])

    return total_images

if __name__ == "__main__":
    with open(LABELS_CSV_PATH, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['image_path', 'class_label', 'field_id', 'capture_date'])

        healthy_count = process_images('healthy', writer)
        blight_count = process_images('lateblt', writer)

    print("\n--------------------")
    print("Curation complete.")
    print(f"Total images processed: {healthy_count + blight_count}")
    print(f"Labels saved to: {LABELS_CSV_PATH}")
    print("--------------------")