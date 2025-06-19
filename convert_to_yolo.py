import os
import json
import yaml
from pathlib import Path

def convert_annotations(json_path, output_dir, img_width, img_height, class_mapping):
    """Convert JSON annotations to YOLO format."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each JSON file
    for json_file in Path(json_path).rglob('*.json'):
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Create output TXT file path
        txt_file = Path(output_dir) / f"{json_file.stem}.txt"
        
        with open(txt_file, 'w') as f_out:
            # Process each object in the image
            for obj in data.get('objects', []):
                class_name = obj['classTitle']
                if class_name not in class_mapping:
                    print(f"Warning: Class '{class_name}' not found in class mapping. Skipping...")
                    continue
                    
                class_id = class_mapping[class_name]
                
                # Convert bounding box to YOLO format (normalized x_center, y_center, width, height)
                bbox = obj['points']['exterior']
                x1, y1 = bbox[0]
                x2, y2 = bbox[1]
                
                # Calculate normalized values
                x_center = ((x1 + x2) / 2) / img_width
                y_center = ((y1 + y2) / 2) / img_height
                width = abs(x2 - x1) / img_width
                height = abs(y2 - y1) / img_height
                
                # Write to file (class_id x_center y_center width height)
                f_out.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

def main():
    # Define paths
    base_dir = "/Users/kshitijtotawar/Desktop/NEW DRONE/dataset-2"
    
    # Class mapping (update this based on your actual class names in JSON)
    class_mapping = {
        "drone": 0,
        "airplane": 1,
        "bird": 2,
        "helicopter": 3,
        "balloon": 4,
        "uav": 5,
        "other": 6
    }
    
    # Image dimensions (from the first image in the dataset)
    img_width = 1024
    img_height = 540
    
    # Process train and test sets
    for split in ['train', 'test']:
        json_dir = os.path.join(base_dir, split, 'ann')
        output_dir = os.path.join(base_dir, split, 'labels')
        
        print(f"Processing {split} set...")
        convert_annotations(json_dir, output_dir, img_width, img_height, class_mapping)
    
    print("Conversion complete!")
    
    # Update data.yaml
    data_yaml = {
        'names': list(class_mapping.keys()),
        'nc': len(class_mapping),
        'path': base_dir,
        'train': 'train/img',
        'val': 'test/img',
        'test': 'test/img'
    }
    
    with open('data.yaml', 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    print("Updated data.yaml configuration")

if __name__ == "__main__":
    main()
