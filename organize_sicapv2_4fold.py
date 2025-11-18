"""
Script to organize SICAPv2 dataset into 4-fold cross-validation structure.

Creates a folder structure:
    data/
    ├── test/          (shared test set - 2,122 patches, 31 slides)
    │   ├── NC/
    │   ├── G3/
    │   ├── G4/
    │   └── G5/
    ├── fold1/
    │   ├── train/     (7,472 patches, 95 slides)
    │   └── valid/     (2,487 patches, 29 slides)
    ├── fold2/
    │   ├── train/     (7,793 patches, 97 slides)
    │   └── valid/     (2,166 patches, 27 slides)
    ├── fold3/
    │   ├── train/     (8,166 patches, 94 slides)
    │   └── valid/     (1,793 patches, 30 slides)
    └── fold4/
        ├── train/     (6,446 patches, 86 slides)
        └── valid/     (3,513 patches, 38 slides)
"""

import os
import shutil
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Paths
SICAP_ROOT = './SICAPv2'  # Path to your SICAPv2 dataset directory on line 34
IMAGES_DIR = os.path.join(SICAP_ROOT, 'images')
OUTPUT_DIR = './data'  # Output directory for organized data on line 36

# Main test set (same for all folds)
TEST_FILE = os.path.join(SICAP_ROOT, 'partition/Test/Test.xlsx')

# Classes
CLASSES = ['NC', 'G3', 'G4', 'G5']


def get_label_from_row(row):
    """Extract the label from a one-hot encoded row."""
    for class_name in CLASSES:
        if row[class_name] == 1:
            return class_name
    return None


def create_folders():
    """Create the folder structure for all 4 folds + test."""
    # Create shared test folder
    for class_name in CLASSES:
        folder_path = os.path.join(OUTPUT_DIR, 'test', class_name)
        os.makedirs(folder_path, exist_ok=True)

    # Create 4 fold folders
    for fold_num in range(1, 5):
        for split in ['train', 'valid']:
            for class_name in CLASSES:
                folder_path = os.path.join(OUTPUT_DIR, f'fold{fold_num}', split, class_name)
                os.makedirs(folder_path, exist_ok=True)

    print("✓ Folder structure created for 4-fold cross-validation")


def copy_images(df, destination_path, split_name):
    """
    Copy images to their respective class folders.

    Args:
        df: DataFrame with image_name and one-hot encoded labels
        destination_path: Base path (e.g., 'data/fold1/train' or 'data/test')
        split_name: Display name for progress bar
    """
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
        dst = os.path.join(destination_path, label, image_name)

        # Copy image if not already exists (avoid duplicate copies)
        if os.path.exists(src):
            if not os.path.exists(dst):  # Only copy if doesn't exist
                shutil.copy2(src, dst)
            copied += 1
        else:
            missing += 1
            if missing <= 5:  # Only print first 5 missing files
                print(f"⚠️  Missing: {image_name}")

    return copied, missing


def print_statistics():
    """Print statistics about the organized dataset."""
    print("\n" + "="*80)
    print("4-FOLD CROSS-VALIDATION DATASET STATISTICS")
    print("="*80)

    # Test set (shared across all folds)
    print("\nSHARED TEST SET:")
    total = 0
    for class_name in CLASSES:
        folder_path = os.path.join(OUTPUT_DIR, 'test', class_name)
        count = len([f for f in os.listdir(folder_path) if f.endswith('.jpg')])
        print(f"  {class_name}: {count:,} images")
        total += count
    print(f"  TOTAL: {total:,} images")

    # Each fold
    for fold_num in range(1, 5):
        print(f"\nFOLD {fold_num}:")
        for split in ['train', 'valid']:
            split_total = 0
            print(f"  {split.upper()}:")
            for class_name in CLASSES:
                folder_path = os.path.join(OUTPUT_DIR, f'fold{fold_num}', split, class_name)
                count = len([f for f in os.listdir(folder_path) if f.endswith('.jpg')])
                print(f"    {class_name}: {count:,} images")
                split_total += count
            print(f"    TOTAL: {split_total:,} images")

    print("\n" + "="*80)


def main():
    """Main function to organize the dataset."""
    print("="*80)
    print("SICAPv2 4-Fold Cross-Validation Data Organization")
    print("="*80)

    # Check if source images exist
    if not os.path.exists(IMAGES_DIR):
        print(f"❌ Error: Images directory not found at {IMAGES_DIR}")
        return

    print(f"📂 Source: {SICAP_ROOT}")
    print(f"📂 Destination: {OUTPUT_DIR}")

    # Create folder structure
    create_folders()

    # Process shared test set
    print("\n📖 Processing shared TEST set...")
    test_df = pd.read_excel(TEST_FILE)
    print(f"✓ Test: {len(test_df)} images")

    test_path = os.path.join(OUTPUT_DIR, 'test')
    copied, missing = copy_images(test_df, test_path, "test")
    print(f"✓ Copied {copied} images to test")
    if missing > 0:
        print(f"⚠️  {missing} images were missing")

    # Process each validation fold
    for fold_num in range(1, 5):
        print(f"\n{'='*80}")
        print(f"📁 Processing FOLD {fold_num}...")
        print(f"{'='*80}")

        val_path = os.path.join(SICAP_ROOT, f'partition/Validation/Val{fold_num}')

        # Load train and validation files for this fold
        train_df = pd.read_excel(os.path.join(val_path, 'Train.xlsx'))
        valid_df = pd.read_excel(os.path.join(val_path, 'Test.xlsx'))

        print(f"✓ Train: {len(train_df)} images")
        print(f"✓ Valid: {len(valid_df)} images")

        # Copy train images
        train_dest = os.path.join(OUTPUT_DIR, f'fold{fold_num}', 'train')
        copied, missing = copy_images(train_df, train_dest, f"fold{fold_num}/train")
        print(f"✓ Copied {copied} images to fold{fold_num}/train")
        if missing > 0:
            print(f"⚠️  {missing} images were missing")

        # Copy validation images
        valid_dest = os.path.join(OUTPUT_DIR, f'fold{fold_num}', 'valid')
        copied, missing = copy_images(valid_df, valid_dest, f"fold{fold_num}/valid")
        print(f"✓ Copied {copied} images to fold{fold_num}/valid")
        if missing > 0:
            print(f"⚠️  {missing} images were missing")

    # Print statistics
    print_statistics()

    print("\n✅ 4-fold cross-validation data organization complete!")
    print(f"\nYou can now train on each fold separately:")
    print(f"  - Fold 1: train on fold1/train, validate on fold1/valid, test on test/")
    print(f"  - Fold 2: train on fold2/train, validate on fold2/valid, test on test/")
    print(f"  - Fold 3: train on fold3/train, validate on fold3/valid, test on test/")
    print(f"  - Fold 4: train on fold4/train, validate on fold4/valid, test on test/")
    print(f"\nThen average the results across all 4 folds for robust performance estimates!")


if __name__ == "__main__":
    main()
