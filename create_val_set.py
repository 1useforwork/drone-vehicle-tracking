import os
import shutil
from sklearn.model_selection import train_test_split

# Set paths
base_dir = "/Users/kshitijtotawar/Desktop/NEW DRONE/datasets"
images_dir = os.path.join(base_dir, "images")
labels_dir = os.path.join(base_dir, "labels")
train_dir = os.path.join(images_dir, "train")
val_dir = os.path.join(images_dir, "val")

# Create val directory if it doesn't exist
os.makedirs(val_dir, exist_ok=True)

# Get list of image files (without extension)
image_files = [f.split('.')[0] for f in os.listdir(train_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

# Split into train and validation (80% train, 20% validation)
train_files, val_files = train_test_split(image_files, test_size=0.2, random_state=42)

print(f"Total images: {len(image_files)}")
print(f"Training set: {len(train_files)} images")
print(f"Validation set: {len(val_files)} images")

# Move validation files to val directory
for file in val_files:
    # Move image
    for ext in ['.jpg', '.jpeg', '.png']:
        src_img = os.path.join(train_dir, f"{file}{ext}")
        if os.path.exists(src_img):
            dst_img = os.path.join(val_dir, f"{file}{ext}")
            shutil.move(src_img, dst_img)
            
            # Move corresponding label
            src_lbl = os.path.join(labels_dir, "train", f"{file}.txt")
            if os.path.exists(src_lbl):
                os.makedirs(os.path.join(labels_dir, "val"), exist_ok=True)
                dst_lbl = os.path.join(labels_dir, "val", f"{file}.txt")
                shutil.move(src_lbl, dst_lbl)
            break

print("\nValidation set created successfully!")
