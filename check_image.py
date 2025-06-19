from PIL import Image
import os

# Check the first image's dimensions
image_path = "/Users/kshitijtotawar/Desktop/NEW DRONE/dataset-2/train/img/M0101_img000001.jpg"
img = Image.open(image_path)
print(f"Image dimensions: {img.size}")  # (width, height)
