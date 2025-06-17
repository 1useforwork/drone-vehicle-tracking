import os
import random
from PIL import Image, ImageDraw

# Configuration
total_images = 1000
train_ratio = 0.9
img_size = 640
classes = ['vehicle']
num_classes = len(classes)

# Paths (change to your absolute path if needed)
base_path = "datasets"
img_train_dir = os.path.join(base_path, "images", "train")
img_val_dir = os.path.join(base_path, "images", "val")
label_train_dir = os.path.join(base_path, "labels", "train")
label_val_dir = os.path.join(base_path, "labels", "val")

# Create directories if not exist
for folder in [img_train_dir, img_val_dir, laimport os
import subprocess

# Paths
synthetic_weights = "runs/detect/train7/weights/best.pt"
real_data_yaml = "real_data.yaml"  # You need to create this YAML pointing to real drone footage annotations
epochs = 50

# Check if synthetic model exists
if os.path.exists(synthetic_weights):
    print("🧠 Synthetic training complete. Starting fine-tuning on real-world data...")

    # Run YOLO training with real-world data
    subprocess.run([
        "yolo", "task=detect", "mode=train",
        f"model={synthetic_weights}",
        f"data={real_data_yaml}",
        f"epochs={epochs}",
        "imgsz=640",
        "name=finetune_real"
    ])
else:
    print("❌ Synthetic model weights not found. Please train the base model first.")
bel_train_dir, label_val_dir]:
    os.makedirs(folder, exist_ok=True)

def create_dummy_image_and_label(img_path, label_path):
    # Create a simple colored background image
    img = Image.new('RGB', (img_size, img_size), color=(
        random.randint(0,255), random.randint(0,255), random.randint(0,255)))

    # Draw random rectangles to simulate bounding boxes (for visualization only)
    draw = ImageDraw.Draw(img)
    num_boxes = random.randint(1, 3)  # each image has 1-3 boxes

    label_lines = []
    for _ in range(num_boxes):
        # Random bbox in pixel coords
        x_min = random.randint(0, img_size - 50)
        y_min = random.randint(0, img_size - 50)
        width = random.randint(30, 100)
        height = random.randint(30, 100)
        x_max = min(x_min + width, img_size)
        y_max = min(y_min + height, img_size)

        # Draw rectangle on image (optional)
        draw.rectangle([x_min, y_min, x_max, y_max], outline="white", width=2)

        # Convert to YOLO format (normalized)
        x_center = ((x_min + x_max) / 2) / img_size
        y_center = ((y_min + y_max) / 2) / img_size
        w_norm = (x_max - x_min) / img_size
        h_norm = (y_max - y_min) / img_size

        class_id = 0  # since only one class 'vehicle'
        label_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

    # Save image
    img.save(img_path)

    # Save label file
    with open(label_path, "w") as f:
        f.write("\n".join(label_lines))

# Create dataset
num_train = int(total_images * train_ratio)
num_val = total_images - num_train

print(f"Creating {num_train} training images and labels...")
for i in range(num_train):
    img_file = os.path.join(img_train_dir, f"img{i}.jpg")
    label_file = os.path.join(label_train_dir, f"img{i}.txt")
    create_dummy_image_and_label(img_file, label_file)

print(f"Creating {num_val} validation images and labels...")
for i in range(num_val):
    img_file = os.path.join(img_val_dir, f"img{i}.jpg")
    label_file = os.path.join(label_val_dir, f"img{i}.txt")
    create_dummy_image_and_label(img_file, label_file)

print("Dummy dataset generation completed.")
