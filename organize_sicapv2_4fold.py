# Script to take the dataset as downloaded and unzipped and create the datasets
# to be used in processing the experiments
#
# References:
#
# Reviewed code from https://github.com/bianhao123 and
# https://github.com/histocartography, https://github.com/cvblab/kd_resolution
# for hints on how to process the files
#
# Creates a folder structure under data with a test subdirectory and one
# subdirectory for each fold.
#

# Here is the output of the run. This is run once and the data is then used
# for all experiments.
#
# Source directory: /Users/salvatorevella/Documents/GitHub/DataScience/CP8321/Project/SICAPv2
# Destination directory: /Users/salvatorevella/Documents/GitHub/DataScience/CP8321/Project/data
# Folder structure created for 4-folds
#
# Start processing shared Test dataset
# Test: 2122 total images
# Copying test (2122 images)...
# Copied 2122 images to test
# Processing Fold 1...
# Train: 7472 images
# Validation: 2487 images
# Copying fold1/train (7472 images)...
# Copied 7472 images to fold1/train
# Copying fold1/valid (2487 images)...
# Copied 2487 images to fold1/valid
# Processing Fold 2...
# Train: 7793 images
# Validation: 2166 images
# Copying fold2/train (7793 images)...
# Copied 7793 images to fold2/train
# Copying fold2/valid (2166 images)...
# Copied 2166 images to fold2/valid
# Processing Fold 3...
# Train: 8166 images
# Validation: 1793 images
# Copying fold3/train (8166 images)...
# Copied 8166 images to fold3/train
# Copying fold3/valid (1793 images)...
# Copied 1793 images to fold3/valid
# Processing Fold 4...
# Train: 6446 images
# Validation: 3513 images
# Copying fold4/train (6446 images)...
# Copied 6446 images to fold4/train
# Copying fold4/valid (3513 images)...
# Copied 3513 images to fold4/valid

import os
import shutil
import pandas as pd
from pathlib import Path

# Paths
SICAP_ROOT = '/Users/salvatorevella/Documents/GitHub/DataScience/CP8321/Project/SICAPv2'
IMAGES_DIR = os.path.join(SICAP_ROOT, 'images')
OUTPUT_DIR = '/Users/salvatorevella/Documents/GitHub/DataScience/CP8321/Project/data'

# Main test set (same for all folds)
TEST_FILE = os.path.join(SICAP_ROOT, 'partition/Test/Test.xlsx')

# Classes
CLASSES = ['NC', 'G3', 'G4', 'G5']

# Helper function ro extract the class label
def get_label_from_row(row):
    for class_name in CLASSES:
        if row[class_name] == 1:
            return class_name
    return None

# Helper function to create the file structure
def create_folders():
    for class_name in CLASSES:
        folder_path = os.path.join(OUTPUT_DIR, 'test', class_name)
        os.makedirs(folder_path, exist_ok=True)

    # Create 4 fold folders
    for fold_num in range(1, 5):
        for split in ['train', 'valid']:
            for class_name in CLASSES:
                folder_path = os.path.join(OUTPUT_DIR, f'fold{fold_num}', split, class_name)
                os.makedirs(folder_path, exist_ok=True)

    print("Folder structure created for 4-folds")

# Copy the images to the proper folder. Note that we check missing but we did
# not actually find any missing - this was defensive and for debugging

def copy_images(df, destination_path, split_name):
    copied = 0
    missing = 0

    print(f"Copying {split_name} ({len(df)} images)...")
    for _, row in df.iterrows():
        image_name = row["image_name"]
        label = get_label_from_row(row)

        if label is None:
            print(f"No label found for {image_name} !!")
            continue

        # Source and destination paths
        src = os.path.join(IMAGES_DIR, image_name)
        dst = os.path.join(destination_path, label, image_name)

        # Copy image if not already exists
        if os.path.exists(src):
            if not os.path.exists(dst):  # Only copy if doesn't exist
                shutil.copy2(src, dst)
            copied += 1
        else:
            missing += 1
            if missing <= 5:  # Print some missing files for debugging
                print(f"Missing: {image_name} !!")

    return copied, missing

def main():
    # Check if source images exist
    if not os.path.exists(IMAGES_DIR):
        print(f"Images directory not found at {IMAGES_DIR} !!")
        return

    print(f"Source directory: {SICAP_ROOT}")
    print(f"Destination directory: {OUTPUT_DIR}")

    # Create folder structure
    create_folders()

    # Process shared test set
    print("\nStart processing shared Test dataset")
    test_df = pd.read_excel(TEST_FILE)
    print(f"Test: {len(test_df)} total images")

    test_path = os.path.join(OUTPUT_DIR, 'test')
    copied, missing = copy_images(test_df, test_path, "test")
    print(f"Copied {copied} images to test")
    if missing > 0:
        print(f"Images were missing - number {missing} ")

    # Process each validation fold
    for fold_num in range(1, 5):
        print(f"Processing Fold {fold_num}...")

        val_path = os.path.join(SICAP_ROOT, f'partition/Validation/Val{fold_num}')

        # Load train and validation files for this fold. These are stored in
        # Excel files
        train_df = pd.read_excel(os.path.join(val_path, 'Train.xlsx'))
        valid_df = pd.read_excel(os.path.join(val_path, 'Test.xlsx'))

        print(f"Train: {len(train_df)} images")
        print(f"Validation: {len(valid_df)} images")

        # Copy Training images
        train_dest = os.path.join(OUTPUT_DIR, f'fold{fold_num}', 'train')
        copied, missing = copy_images(train_df, train_dest, f"fold{fold_num}/train")
        print(f"Copied {copied} images to fold{fold_num}/train")

        if missing > 0:
            print(f"{missing} images were missing")

        # Copy Validation images
        valid_dest = os.path.join(OUTPUT_DIR, f'fold{fold_num}', 'valid')
        copied, missing = copy_images(valid_df, valid_dest, f"fold{fold_num}/valid")
        print(f"Copied {copied} images to fold{fold_num}/valid")

        if missing > 0:
            print(f"{missing} images were missing")

if __name__ == "__main__":
    main()
