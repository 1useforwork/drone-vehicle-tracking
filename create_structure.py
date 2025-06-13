import os

# Root directory
root = "Drone intern"
os.makedirs(root, exist_ok=True)

# Python script files
script_files = ["detect.py", "track.py", "density.py", "main.py", "data.yaml"]
for file in script_files:
    open(os.path.join(root, file), 'a').close()

# Dataset directories
image_train_path = os.path.join(root, "datasets/images/train")
image_val_path = os.path.join(root, "datasets/images/val")
label_train_path = os.path.join(root, "datasets/labels/train")
label_val_path = os.path.join(root, "datasets/labels/val")

for path in [image_train_path, image_val_path, label_train_path, label_val_path]:
    os.makedirs(path, exist_ok=True)

# Sample placeholder files (optional)
open(os.path.join(image_train_path, "image1.jpg"), 'a').close()
open(os.path.join(image_val_path, "image2.jpg"), 'a').close()
open(os.path.join(label_train_path, "image1.txt"), 'a').close()
open(os.path.join(label_val_path, "image2.txt"), 'a').close()

print(f"✅ Folder structure created under '{root}'")
