import os
import glob
import csv

# --- CONFIGURATION ---
IMAGES_BASE_PATH = "data/images"
LABELS_CSV_PATH = "data/labels.csv"

def rebuild_labels_file():
    """
    Scans train/val/test directories and rebuilds the labels.csv file
    based on the filenames.
    """
    print(f"Starting to rebuild {LABELS_CSV_PATH}...")

    # This will hold all the data before writing
    all_records = []

    # The folders to scan
    split_folders = ["train", "val", "test"]

    for split in split_folders:
        folder_path = os.path.join(IMAGES_BASE_PATH, split)
        image_files = glob.glob(os.path.join(folder_path, '*.jpg'))

        print(f"Found {len(image_files)} images in '{split}' folder.")

        for img_path in image_files:
            filename = os.path.basename(img_path)

            # Determine the class from the filename
            class_label = "unknown"
            if filename.lower().startswith('healthy'):
                class_label = 'healthy'
            elif filename.lower().startswith('lateblt'): # Assuming this is the prefix for late blight
                class_label = 'lateblt'

            # Use forward slashes for the path in the CSV
            path_for_csv = img_path.replace('\\', '/')

            all_records.append([path_for_csv, class_label, 'tanzania_field_1', '2024-01-01'])

    # Write all collected records to the CSV file
    with open(LABELS_CSV_PATH, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['image_path', 'class_label', 'field_id', 'capture_date'])
        writer.writerows(all_records)

    print(f"\nSuccessfully rebuilt labels.csv with {len(all_records)} records.")

if __name__ == "__main__":
    rebuild_labels_file()