# =============================================================
# STEP 2: CLEAN & SPLIT DATASET
# Role: Data Engineer
# Description: Remove corrupt images, split into train/val/test
# =============================================================

import os
import shutil
from PIL import Image
from sklearn.model_selection import train_test_split

RAW_DIR = "shark_dataset/raw"
SPLIT_DIR = "shark_dataset/split"

# Create split folders
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(SPLIT_DIR, split), exist_ok=True)

total_valid = 0
total_removed = 0

for class_name in sorted(os.listdir(RAW_DIR)):
    class_path = os.path.join(RAW_DIR, class_name)
    if not os.path.isdir(class_path):
        continue

    print(f"\n Processing class: {class_name}")

    # Filter out corrupt images
    valid_images = []
    for fname in os.listdir(class_path):
        fpath = os.path.join(class_path, fname)
        try:
            with Image.open(fpath) as img:
                img.verify()
            valid_images.append(fpath)
        except Exception:
            print(f"  Removing corrupt file: {fname}")
            os.remove(fpath)
            total_removed += 1

    # Split: 70% train, 15% val, 15% test
    train_files, temp = train_test_split(valid_images, test_size=0.30, random_state=42)
    val_files, test_files = train_test_split(temp, test_size=0.50, random_state=42)

    for split_name, files in zip(["train", "val", "test"], [train_files, val_files, test_files]):
        dest_dir = os.path.join(SPLIT_DIR, split_name, class_name)
        os.makedirs(dest_dir, exist_ok=True)
        for f in files:
            shutil.copy(f, dest_dir)

    total_valid += len(valid_images)
    print(f"  Train: {len(train_files)} | Val: {len(val_files)} | Test: {len(test_files)}")

print(f"\n Dataset split complete!")
print(f"  Total valid images : {total_valid}")
print(f"  Total removed      : {total_removed}")
print(f"  Saved to           : {SPLIT_DIR}")
