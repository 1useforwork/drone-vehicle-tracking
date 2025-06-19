import os
import shutil
import random
from pathlib import Path

# Set random seed for reproducibility
random.seed(42)

# Paths
data_dir = Path("/Users/kshitijtotawar/Desktop/NEW DRONE/dataset-2")
train_img_dir = data_dir / "train/images"
train_label_dir = data_dir / "labels/train"
val_img_dir = data_dir / "test/images"
val_label_dir = data_dir / "labels/val"

# Create validation directories if they don't exist
val_img_dir.mkdir(parents=True, exist_ok=True)
val_label_dir.mkdir(parents=True, exist_ok=True)

# Get list of all training images
train_images = list(train_img_dir.glob("*.jpg"))
print(f"Found {len(train_images)} training images")

# Use 20% of training data for validation
val_size = int(0.2 * len(train_images))
val_images = random.sample(train_images, val_size)
print(f"Moving {len(val_images)} images to validation set")

# Move files to validation set
for img_path in val_images:
    # Move image
    dest_img = val_img_dir / img_path.name
    shutil.move(str(img_path), str(dest_img))
    
    # Move corresponding label
    label_name = img_path.stem + ".txt"
    src_label = train_label_dir / label_name
    if src_label.exists():
        dest_label = val_label_dir / label_name
        shutil.move(str(src_label), str(dest_label))
    else:
        print(f"Warning: Missing label for {img_path.name}")

print("Dataset split complete!")
print(f"Training images: {len(list(train_img_dir.glob('*.jpg')))}")
print(f"Validation images: {len(list(val_img_dir.glob('*.jpg')))}")
