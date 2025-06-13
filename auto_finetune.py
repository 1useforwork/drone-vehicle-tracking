import os
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
