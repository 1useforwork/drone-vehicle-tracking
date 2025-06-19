import os
import yaml

# Set your dataset path
dataset_dir = "datasets"
images_dir = os.path.join(dataset_dir, "images")

# Auto-detect train/val folders
train_dir = os.path.join(images_dir, "train")
val_dir = os.path.join(images_dir, "val")

# Function to infer class names from label files
def infer_classes(label_dir):
    classes = set()
    for root, _, files in os.walk(label_dir):
        for file in files:
            if file.endswith(".txt"):
                with open(os.path.join(root, file), "r") as f:
                    for line in f:
                        if line.strip():
                            class_id = int(line.split()[0])
                            classes.add(class_id)
    return sorted(list(classes))

# Infer number of classes from label files
labels_dir = os.path.join(dataset_dir, "labels", "train")
class_ids = infer_classes(labels_dir)
nc = len(class_ids)
names = [f"class_{i}" for i in class_ids]

# Output YAML structure
data_yaml = {
    "path": dataset_dir,
    "train": os.path.relpath(train_dir, dataset_dir),
    "val": os.path.relpath(val_dir, dataset_dir),
    "nc": nc,
    "names": names
}

# Save YAML
with open("data.yaml", "w") as f:
    yaml.dump(data_yaml, f)

print("✅ data.yaml generated successfully with:")
print(yaml.dump(data_yaml))