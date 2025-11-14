"""
Script to organize SICAPv2 dataset into train/valid/test folders for CNN training.

This script reads the Excel partition files and creates a folder structure:
    data/
    ├── train/
    │   ├── NC/
    │   ├── G3/
    │   ├── G4/
    │   └── G5/
    ├── valid/
    │   └── (same structure)
    └── test/
        └── (same structure)
"""

import os
import shutil
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Paths
SICAP_ROOT = '/Users/salvatorevella/Documents/GitHub/CP8321/SICAPv2'
IMAGES_DIR = os.path.join(SICAP_ROOT, 'images')
OUTPUT_DIR = '/Users/salvatorevella/Documents/GitHub/CP8321/data'

# Partition files
TRAIN_FILE = os.path.join(SICAP_ROOT, 'partition/Test/Train.xlsx')
TEST_FILE = os.path.join(SICAP_ROOT, 'partition/Test/Test.xlsx')
VALID_FILE = os.path.join(SICAP_ROOT, 'partition/Validation/Val1/Test.xlsx')  # Use Val1 test as validation

# Classes
CLASSES = ['NC', 'G3', 'G4', 'G5']


def get_label_from_row(row):
    """
    Extract the label from a one-hot encoded row.

    Args:
        row: DataFrame row with columns NC, G3, G4, G5

    Returns:
        str: Label name (NC, G3, G4, or G5)
    """
    for class_name in CLASSES:
        if row[class_name] == 1:
            return class_name
    return None


def create_folders():
    """Create the folder structure for train/valid/test."""
    for split in ['train', 'valid', 'test']:
        for class_name in CLASSES:
            folder_path = os.path.join(OUTPUT_DIR, split, class_name)
            os.makedirs(folder_path, exist_ok=True)
    print("✓ Folder structure created")


def copy_images(df, split_name):
    """
    Copy images to their respective class folders.

    Args:
        df: DataFrame with image_name and one-hot encoded labels
        split_name: 'train', 'valid', or 'test'
    """
    print(f"\n📁 Processing {split_name} set...")
    copied = 0
    missing = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Copying {split_name}"):
        image_name = row['image_name']
        label = get_label_from_row(row)

        if label is None:
            print(f"⚠️  Warning: No label found for {image_name}")
            continue

        # Source and destination paths
        src = os.path.join(IMAGES_DIR, image_name)
        dst = os.path.join(OUTPUT_DIR, split_name, label, image_name)

        # Copy image
        if os.path.exists(src):
            shutil.copy2(src, dst)
            copied += 1
        else:
            missing += 1
            if missing <= 5:  # Only print first 5 missing files
                print(f"⚠️  Missing: {image_name}")

    print(f"✓ Copied {copied} images to {split_name}")
    if missing > 0:
        print(f"⚠️  {missing} images were missing")


def print_statistics():
    """Print statistics about the organized dataset."""
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)

    for split in ['train', 'valid', 'test']:
        print(f"\n{split.upper()}:")
        total = 0
        for class_name in CLASSES:
            folder_path = os.path.join(OUTPUT_DIR, split, class_name)
            count = len([f for f in os.listdir(folder_path) if f.endswith('.jpg')])
            print(f"  {class_name}: {count:,} images")
            total += count
        print(f"  TOTAL: {total:,} images")

    print("\n" + "="*60)


def main():
    """Main function to organize the dataset."""
    print("="*60)
    print("SICAPv2 Data Organization Script")
    print("="*60)

    # Check if source images exist
    if not os.path.exists(IMAGES_DIR):
        print(f"❌ Error: Images directory not found at {IMAGES_DIR}")
        return

    print(f"📂 Source: {SICAP_ROOT}")
    print(f"📂 Destination: {OUTPUT_DIR}")

    # Create folder structure
    create_folders()

    # Load partition files
    print("\n📖 Loading partition files...")
    train_df = pd.read_excel(TRAIN_FILE)
    test_df = pd.read_excel(TEST_FILE)
    valid_df = pd.read_excel(VALID_FILE)

    print(f"✓ Train: {len(train_df)} images")
    print(f"✓ Valid: {len(valid_df)} images")
    print(f"✓ Test: {len(test_df)} images")

    # Copy images to their folders
    copy_images(train_df, 'train')
    copy_images(valid_df, 'valid')
    copy_images(test_df, 'test')

    # Print statistics
    print_statistics()

    print("\n✅ Data organization complete!")
    print(f"\nYou can now use the data from: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
