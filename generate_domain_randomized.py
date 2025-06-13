import cv2
import numpy as np
import os
import random
from tqdm import tqdm

# Configuration
output_dir = "datasets/images/train"
label_dir = "datasets/labels/train"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(label_dir, exist_ok=True)

def random_background():
    bg_color = np.random.randint(150, 255, size=3)
    return np.full((640, 640, 3), bg_color, dtype=np.uint8)

def add_random_vehicle(img):
    # Simulate a vehicle as a red rectangle
    x1, y1 = random.randint(50, 500), random.randint(50, 500)
    w, h = random.randint(40, 100), random.randint(20, 60)
    x2, y2 = x1 + w, y1 + h
    color = (0, 0, 255)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
    return img, x1, y1, x2, y2

def apply_random_effects(img):
    if random.random() < 0.3:
        img = cv2.GaussianBlur(img, (5, 5), 0)
    if random.random() < 0.3:
        img = cv2.convertScaleAbs(img, alpha=random.uniform(0.5, 1.5), beta=random.randint(-50, 50))
    return img

def simulate_thermal_signature(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thermal = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    return thermal

# Generate 1,000 synthetic images
for i in tqdm(range(1000)):
    img = random_background()
    img, x1, y1, x2, y2 = add_random_vehicle(img)
    img = apply_random_effects(img)

    # 🔥 Optional: simulate heat signature view (enabled 50% of the time)
    if random.random() < 0.5:
        img = simulate_thermal_signature(img)

    filename = f"{i}.jpg"
    cv2.imwrite(os.path.join(output_dir, filename), img)

    # YOLO label format: class x_center y_center width height (normalized)
    xc = (x1 + x2) / 2 / 640
    yc = (y1 + y2) / 2 / 640
    w = (x2 - x1) / 640
    h = (y2 - y1) / 640

    with open(os.path.join(label_dir, f"{i}.txt"), "w") as f:
        f.write(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
