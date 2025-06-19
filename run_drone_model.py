import os
import torch
from ultralytics import YOLO
import cv2
import numpy as np

def main():
    # Check if CUDA (GPU) is available
    device = 'cuda' if torch.cuda.is_available() or torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load the model
    model_path = "best_drone_model.pt"
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    
    # Move model to device (GPU if available, otherwise CPU)
    model.to(device)
    
    # Path to your test images
    test_images_dir = "../datasets/images/train"  # Using training images for demonstration
    
    # Check if test directory exists
    if not os.path.exists(test_images_dir):
        print(f"Test directory not found: {test_images_dir}")
        print("Please make sure the datasets are properly set up.")
        return
    
    # Get list of images (using first 5 images for demonstration)
    image_files = sorted([f for f in os.listdir(test_images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    image_files = image_files[:5]  # Process only first 5 images for demonstration
    
    if not image_files:
        print(f"No images found in {test_images_dir}")
        return
    
    print(f"Found {len(image_files)} images for testing")
    
    # Create output directory
    os.makedirs('output', exist_ok=True)
    
    # Process each image
    for i, img_file in enumerate(image_files, 1):
        img_path = os.path.join(test_images_dir, img_file)
        print(f"\nProcessing image {i}/{len(image_files)}: {img_file}")
        
        # Read image
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"Failed to load image: {img_path}")
            continue
        
        # Run detection
        print("Running detection...")
        results = model(frame)
        
        # Visualize results
        print("Visualizing results...")
        annotated_frame = results[0].plot()
        
        # Save output
        output_path = os.path.join('output', f'result_{img_file}')
        cv2.imwrite(output_path, annotated_frame)
        print(f"Saved result to: {output_path}")
    
    print("\nProcessing complete! Check the 'output' directory for results.")

if __name__ == "__main__":
    main()
