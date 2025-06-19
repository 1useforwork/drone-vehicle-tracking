import os
import torch
from ultralytics import YOLO
import yaml

def train_model():
    # Check device availability
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model (using YOLOv8n as base)
    model = YOLO('yolov8n.pt')  # or yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
    
    # Train the model with 100 epochs and optimized parameters
    results = model.train(
        data='data.yaml',
        epochs=100,  # Increased for better convergence
        patience=25,  # Early stopping if no improvement for 25 epochs
        batch=8,     # Batch size
        imgsz=640,  # Image size
        device=device,  # Use MPS (Metal) on Mac
        workers=8,  # Number of worker threads
        project='drone_detection',
        name='yolov8n_drone_enhanced',
        exist_ok=True,
        pretrained=True,  # Use pretrained weights
        optimizer='AdamW',
        lr0=0.01,  # Initial learning rate
        lrf=0.01,  # Final learning rate (lr0 * lrf)
        momentum=0.9,
        weight_decay=0.0005,
        warmup_epochs=5.0,  # Warmup epochs
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,  # Box loss gain
        cls=0.5,  # Class loss gain
        dfl=1.5,  # Distribution Focal Loss gain
        hsv_h=0.015,  # Image HSV-Hue augmentation (fraction)
        hsv_s=0.7,  # Image HSV-Saturation augmentation (fraction)
        hsv_v=0.4,  # Image HSV-Value augmentation (fraction)
        degrees=10.0,  # Image rotation (+/- deg)
        translate=0.1,  # Image translation (+/- fraction)
        scale=0.5,  # Image scale (+/- gain)
        shear=2.0,  # Image shear (+/- deg)
        perspective=0.0,  # Image perspective (+/- fraction)
        flipud=0.0,  # Image flip up-down (probability)
        fliplr=0.5,  # Image flip left-right (probability)
        mosaic=1.0,  # Image mosaic (probability)
        mixup=0.1,  # Image mixup (probability)
        copy_paste=0.0,  # Segment copy-paste (probability)
        auto_augment='randaugment',  # Auto-augment policy
        erasing=0.4,  # Random erasing (probability)
        cache='ram',  # Cache images in RAM for faster training
        single_cls=False,  # Treat as single-class dataset
        augment=True,  # Apply image augmentation
        verbose=True,  # Print results per class
        seed=42,  # Global training seed
        deterministic=True,  # Reproducible training
        plots=True,  # Save plots during training
        save_period=5,  # Save checkpoint every 5 epochs
        val=True  # Validate during training
    )

if __name__ == "__main__":
    train_model()
