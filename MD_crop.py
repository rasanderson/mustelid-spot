#!/usr/bin/env python3
"""
Animal Detector and Cropper
This script uses MegaDetectorV6 via PytorchWildlife to detect animals in images and crop them.
Usage: python animal_detector.py /path/to/image/folder --output /path/to/output/folder
"""

import os
import cv2
import torch
import argparse
from pathlib import Path
from tqdm import tqdm
from PytorchWildlife.models import detection as pw_detection

# Global variable to store image mapping
global_folder_images = {}

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Detect and crop animals in images.')
    parser.add_argument('folder', type=str, help='Path to folder containing images')
    parser.add_argument('--output', type=str, help='Output folder for cropped images (default: same as original image folder)')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for detection')
    parser.add_argument('--threshold', type=float, default=0.5, help='Confidence threshold for detections')
    parser.add_argument('--recursive', action='store_true', help='Process images in subfolders recursively')
    return parser.parse_args()

def process_detection_results(results, threshold=0.5, output_folder=None):
    """Process detection results and save crops for animal detections."""
    processed_count = 0
    
    for result in results:
        img_path = Path(result['img_id'])
        detections = result['detections']
        
        # Skip if no detections
        if detections is None:
            continue
            
        # Get the original image path from the filename
        # The batch_image_detection function returns paths in the temp directory
        # We need to get the original file path to save crops in the right place
        basename = os.path.basename(str(img_path))
        
        # Find the corresponding original image - we'll search all images to process
        original_img_path = None
        # This will be set by the calling function
        for folder, imgs in global_folder_images.items():
            for img in imgs:
                if os.path.basename(str(img)) == basename:
                    original_img_path = img
                    break
            if original_img_path:
                break
                
        if not original_img_path:
            print(f"Warning: Could not find original path for {img_path}, using detected path")
            original_img_path = img_path
        else:
            print(f"Found original path: {original_img_path} for {basename}")
            
        # Now load the image from the original location
        image = cv2.imread(str(original_img_path))
        if image is None:
            print(f"Error: Could not read image {original_img_path}")
            continue
            
        height, width = image.shape[:2]
        animal_count = 0
        
        # Determine output directory
        if output_folder:
            output_dir = Path(output_folder)
            # Create output directory if it doesn't exist
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = original_img_path.parent
        
        # Handle different result formats
        if hasattr(detections, 'xyxy'):
            # Original format with detections object
            if len(detections.xyxy) == 0:
                continue
                
            # Get detection data
            boxes = detections.xyxy
            confidences = detections.confidence
            class_ids = detections.class_id
            
            # Process each detection
            for i in range(len(boxes)):
                # Class 0 is animal in MegaDetectorV6-yolov10-c
                if class_ids[i] == 0 and confidences[i] >= threshold: # Can change for detecting humans (class == 1), or vehicles (class == 2)
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = boxes[i]
                    
                    # Convert to integers and ensure valid dimensions
                    x1, y1 = max(0, int(x1)), max(0, int(y1))
                    x2, y2 = min(width, int(x2)), min(height, int(y2))
                    
                    # Skip if dimensions are too small
                    if x2 - x1 < 10 or y2 - y1 < 10:
                        continue
                    
                    # Crop the image
                    cropped_image = image[y1:y2, x1:x2]
                    
                    # Generate output path with original filename
                    output_filename = original_img_path.stem + "_crop"
                    if animal_count > 0:
                        output_filename += f"_{animal_count}"
                    output_filename += original_img_path.suffix
                    output_path = output_dir / output_filename
                    
                    # Handle filename conflicts by adding numbers
                    counter = 1
                    while output_path.exists():
                        conflict_filename = original_img_path.stem + "_crop"
                        if animal_count > 0:
                            conflict_filename += f"_{animal_count}"
                        conflict_filename += f"_({counter})" + original_img_path.suffix
                        output_path = output_dir / conflict_filename
                        counter += 1
                    
                    # Save the cropped image
                    cv2.imwrite(str(output_path), cropped_image)
                    print(f"Saved animal crop to {output_path}")
                    animal_count += 1
                    processed_count += 1
        elif isinstance(detections, dict):
            # Handle dictionary format returned by image_detection
            # Check if there are any animal detections
            if 'boxes' in detections and len(detections['boxes']) > 0:
                boxes = detections['boxes']
                scores = detections['scores'] if 'scores' in detections else []
                labels = detections['labels'] if 'labels' in detections else []
                
                for i in range(len(boxes)):
                    # Check if this is an animal with sufficient confidence
                    is_animal = (i < len(labels) and labels[i] == 0) or len(labels) == 0
                    has_confidence = (i < len(scores) and scores[i] >= threshold) or len(scores) == 0
                    
                    if is_animal and has_confidence:
                        # Get bounding box coordinates
                        if len(boxes[i]) == 4:
                            x1, y1, x2, y2 = boxes[i]
                            
                            # Convert to integers and ensure valid dimensions
                            x1, y1 = max(0, int(x1)), max(0, int(y1))
                            x2, y2 = min(width, int(x2)), min(height, int(y2))
                            
                            # Skip if dimensions are too small
                            if x2 - x1 < 10 or y2 - y1 < 10:
                                continue
                            
                            # Crop the image
                            cropped_image = image[y1:y2, x1:x2]
                            
                            # Generate output path with original filename
                            output_filename = original_img_path.stem + "_crop"
                            if animal_count > 0:
                                output_filename += f"_{animal_count}"
                            output_filename += original_img_path.suffix
                            output_path = output_dir / output_filename
                            
                            # Handle filename conflicts by adding numbers
                            counter = 1
                            while output_path.exists():
                                conflict_filename = original_img_path.stem + "_crop"
                                if animal_count > 0:
                                    conflict_filename += f"_{animal_count}"
                                conflict_filename += f"_({counter})" + original_img_path.suffix
                                output_path = output_dir / conflict_filename
                                counter += 1
                            
                            # Save the cropped image
                            cv2.imwrite(str(output_path), cropped_image)
                            print(f"Saved animal crop to {output_path}")
                            animal_count += 1
                            processed_count += 1
    
    return processed_count

def find_processable_images(folder, recursive=False, output_folder=None):
    """
    Find all image files that should be processed.
    
    Conditions for processing:
    1. File is an image with valid extension
    2. Filename does not contain 'crop'
    3. There is no existing crop version of the file (checks output folder if specified)
    """
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    folder_path = Path(folder)
    
    # Define the glob pattern based on recursion setting
    pattern_prefix = '**/*' if recursive else '*'
    
    # Find all images
    all_images = []
    for ext in image_extensions:
        all_images.extend(list(folder_path.glob(f'{pattern_prefix}{ext}')))
        all_images.extend(list(folder_path.glob(f'{pattern_prefix}{ext.upper()}')))
    
    # Create a set of existing crops for quick lookup
    existing_crops = set()
    
    if output_folder:
        # Check for existing crops in the output folder
        output_path = Path(output_folder)
        if output_path.exists():
            for ext in image_extensions:
                existing_crops.update(str(img.stem) for img in output_path.glob(f'*{ext}'))
                existing_crops.update(str(img.stem) for img in output_path.glob(f'*{ext.upper()}'))
    else:
        # Check for existing crops in the same directories as the original images
        for ext in image_extensions:
            existing_crops.update(str(img.stem) for img in folder_path.glob(f'{pattern_prefix}{ext}'))
            existing_crops.update(str(img.stem) for img in folder_path.glob(f'{pattern_prefix}{ext.upper()}'))
    
    # Find images that need processing
    images_to_process = []
    for img in all_images:
        # Skip if 'crop' is in the filename
        if 'crop' in img.stem:
            continue
            
        # Check if a crop version exists
        crop_exists = False
        base_crop_name = f"{img.stem}_crop"
        
        # Check for base crop or numbered crops
        if base_crop_name in existing_crops:
            crop_exists = True
        else:
            # Check for numbered crops (_crop_0, _crop_1, etc.)
            for i in range(10):  # Check reasonable number of crops
                numbered_crop = f"{img.stem}_crop_{i}"
                if numbered_crop in existing_crops:
                    crop_exists = True
                    break
        
        # If no crop exists, add to processing list
        if not crop_exists:
            images_to_process.append(img)
    
    return sorted(images_to_process)

def process_folder(folder_path, images, detection_model, batch_size, threshold, output_folder=None):
    """Process a single folder of images."""
    print(f"Processing folder: {folder_path}")
    print(f"Found {len(images)} images needing processing")
    
    if output_folder:
        print(f"Output folder: {output_folder}")
    
    if not images:
        return 0
    
    # Create temporary directory with symlinks to images to process
    import tempfile
    import shutil
    
    total_processed = 0
    batch_count = (len(images) + batch_size - 1) // batch_size
    
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        batch_num = i // batch_size + 1
        print(f"Processing batch {batch_num}/{batch_count} ({len(batch)} images)")
        
        # Create a temporary directory for this batch
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create symbolic links to all images in this batch
            for img_path in batch:
                # Create a symbolic link in the temp directory
                link_path = os.path.join(temp_dir, os.path.basename(str(img_path)))
                try:
                    # On Windows, might need special permissions for symlinks
                    if os.name == 'nt':  # Windows
                        # Instead of symlink, just copy the file
                        shutil.copy2(str(img_path), link_path)
                    else:  # Unix-like
                        os.symlink(str(img_path), link_path)
                except Exception as e:
                    print(f"Warning: Failed to create link/copy for {img_path}: {e}")
                    # As a fallback, copy the file
                    try:
                        shutil.copy2(str(img_path), link_path)
                    except Exception as e2:
                        print(f"Error: Could not copy {img_path}: {e2}")
                        continue
            
            try:
                # Run detection on this temporary directory
                results = detection_model.batch_image_detection(
                    temp_dir, 
                    batch_size=len(batch)
                )
                
                # Process and save crops
                crops = process_detection_results(results, threshold, output_folder)
                total_processed += crops
                
            except Exception as e:
                print(f"Error processing batch: {e}")
                import traceback
                traceback.print_exc()
    
    return total_processed

def main():
    global global_folder_images
    
    args = parse_args()
    
    # Validate folder exists
    if not os.path.exists(args.folder):
        print(f"Error: Folder {args.folder} does not exist")
        return
    
    # Validate output folder if specified
    if args.output:
        output_path = Path(args.output)
        try:
            output_path.mkdir(parents=True, exist_ok=True)
            print(f"Output folder: {args.output}")
        except Exception as e:
            print(f"Error: Could not create output folder {args.output}: {e}")
            return
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print("Initializing MegaDetector...")
    detection_model = pw_detection.MegaDetectorV6(device=device, pretrained=True, version="MDV6-yolov10-c")
    
    # First, get all images that need processing
    print("Finding images that need processing...")
    image_files = find_processable_images(args.folder, args.recursive, args.output)
    print(f"Found {len(image_files)} images to process")
    
    if not image_files:
        print("No images found that need processing. Exiting.")
        return
    
    # Group images by folder to process one folder at a time
    global_folder_images = {}
    for img in image_files:
        folder = img.parent
        if folder not in global_folder_images:
            global_folder_images[folder] = []
        global_folder_images[folder].append(img)
    
    # Process each folder
    total_crops = 0
    for folder_idx, (folder, images) in enumerate(global_folder_images.items(), 1):
        print(f"\n[{folder_idx}/{len(global_folder_images)}] Processing folder: {folder}")
        crops = process_folder(folder, images, detection_model, args.batch_size, args.threshold, args.output)
        total_crops += crops
    
    print(f"\nProcessing complete! Created {total_crops} animal crops.")
    if args.output:
        print(f"All crops saved to: {args.output}")

if __name__ == "__main__":
    main()