# =============================================================
# STEP 3: STANDARDIZE DATASET
# Role: Data Engineer
# Description: Resize all images to 224x224, convert to RGB JPEG
# =============================================================

import os
from PIL import Image

SPLIT_DIR = "shark_dataset/split"
IMG_SIZE = (224, 224)  # Standard input size for ResNet50, DenseNet121, MobileNetV3

splits = ["train", "val", "test"]
total_processed = 0
total_errors = 0

for split in splits:
    split_path = os.path.join(SPLIT_DIR, split)
    print(f"\n Standardizing [{split}] set...")

    for class_name in sorted(os.listdir(split_path)):
        class_path = os.path.join(split_path, class_name)
        if not os.path.isdir(class_path):
            continue

        count = 0
        for fname in os.listdir(class_path):
            fpath = os.path.join(class_path, fname)
            try:
                with Image.open(fpath) as img:
                    # Convert all modes to RGB (handles RGBA, grayscale, palette)
                    img = img.convert("RGB")
                    # Resize to 224x224 using high-quality Lanczos filter
                    img = img.resize(IMG_SIZE, Image.LANCZOS)
                    # Save as JPEG (rename file if extension is different)
                    new_fname = os.path.splitext(fname)[0] + ".jpg"
                    new_fpath = os.path.join(class_path, new_fname)
                    img.save(new_fpath, "JPEG", quality=95)
                    # Remove old file if extension changed
                    if new_fpath != fpath:
                        os.remove(fpath)
                    count += 1
                    total_processed += 1
            except Exception as e:
                print(f"  Error on {fname}: {e}")
                try:
                    os.remove(fpath)
                except:
                    pass
                total_errors += 1

        print(f"  {class_name}: {count} images standardized")

print(f"\n Standardization complete!")
print(f"  Total processed : {total_processed}")
print(f"  Total errors    : {total_errors}")
print(f"  Image size      : {IMG_SIZE[0]}x{IMG_SIZE[1]} RGB JPEG")
