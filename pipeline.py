import os
import argparse
import csv
import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
import tensorflow as tf
from tqdm import tqdm
from PytorchWildlife.models import detection as pw_detection
import PytorchWildlife.utils as pw_utils
import json
import shutil

# Fixed paths and settings
MODEL_NAME = "fox_v8_20250307_170508"
MODEL_PATH = f"C:/Users/c0062193.CAMPUS/OneDrive - Newcastle University/General - Fox-AI/AI results/Model performance/{MODEL_NAME}/models/best_model_{MODEL_NAME}.h5"
CLASS_NAMES = ['fox', 'lagomorph', 'person', 'squirrel', 'badger', 'dog']
TARGET_SIZE = (224, 224)
DETECTION_THRESHOLD = 0.3
CLASSIFICATION_THRESHOLD = 0.5
BATCH_SIZE = 16

def load_classification_model():
    """Load the classification model"""
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"Classification model loaded successfully from {MODEL_PATH}")
        return model
    except Exception as e:
        print(f"Error loading classification model: {e}")
        return None

def classify_images(crop_dir, model, output_csv):
    """Classify cropped images and save results to CSV"""
    # Get all cropped images
    image_files = []
    for root, _, files in os.walk(crop_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_files.append(os.path.join(root, file))
    
    print(f"Found {len(image_files)} cropped images to classify")
    
    # Prepare CSV file
    with open(output_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['crop_file', 'class', 'confidence'])
        
        # Process each image
        for img_path in tqdm(image_files):
            try:
                # Load and preprocess image
                img = tf.keras.preprocessing.image.load_img(
                    img_path, target_size=TARGET_SIZE, color_mode='grayscale')
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                img_array = img_array / 255.0  # Normalize to [0,1]
                img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
                
                # Add channel dimension for grayscale
                img_array = np.expand_dims(img_array, axis=-1)
                
                # Make prediction
                pred = model.predict(img_array, verbose=0)
                
                # Process prediction (handles both binary and multi-class)
                if pred.shape[1] > 1:  # Multi-class
                    pred_class_idx = np.argmax(pred[0])
                    confidence = float(pred[0][pred_class_idx])
                else:  # Binary
                    confidence = float(pred[0][0])
                    pred_class_idx = 1 if confidence > 0.5 else 0
                
                # Get class name
                pred_class = "Unknown (low confidence)"
                if confidence >= CLASSIFICATION_THRESHOLD:
                    pred_class = CLASS_NAMES[pred_class_idx] if pred_class_idx < len(CLASS_NAMES) else f"Class {pred_class_idx}"
                
                # Write result to CSV
                csv_writer.writerow([os.path.basename(img_path), pred_class, confidence])
                
            except Exception as e:
                print(f"Error classifying {img_path}: {e}")
                csv_writer.writerow([os.path.basename(img_path), "Error", ""])
    
    print(f"Classification results saved to {output_csv}")

def organize_by_megadetector_class(detection_results, output_dir):
    """
    Organize images into folders based on MegaDetector classification:
    - empty: no detections
    - vehicle: contains vehicle detections
    - person: contains person detections
    - animal: contains animal detections
    
    Returns the path to the animal folder.
    """
    # Create class-specific folders
    empty_dir = os.path.join(output_dir, 'empty')
    vehicle_dir = os.path.join(output_dir, 'vehicle')
    person_dir = os.path.join(output_dir, 'person')
    animal_dir = os.path.join(output_dir, 'animal')
    
    for folder in [empty_dir, vehicle_dir, person_dir, animal_dir]:
        os.makedirs(folder, exist_ok=True)
    
    # MegaDetector class mapping (based on MegaDetectorV6)
    # 0: animal, 1: person, 2: vehicle
    class_folders = {
        0: animal_dir,
        1: person_dir,
        2: vehicle_dir
    }
    
    # Process each image
    for img_result in detection_results:
        # Extract the image path - the structure is different in PytorchWildlife
        if 'image_id' in img_result:
            img_path = img_result['image_id']  # Use image_id which contains the file path
        elif 'filename' in img_result:
            img_path = img_result['filename']
        else:
            print(f"Warning: Could not determine file path for result: {img_result}")
            continue
            
        img_filename = os.path.basename(img_path)
        
        # Default to empty if no detections above threshold
        target_folder = empty_dir
        
        # Check if there are any detections
        if 'detections' in img_result and img_result['detections']:
            # Get the highest confidence detection
            valid_detections = [det for det in img_result['detections'] 
                               if det.get('conf', 0) >= DETECTION_THRESHOLD]
            
            if valid_detections:
                # Sort by category_id priority: animal (0), person (1), vehicle (2)
                # This ensures if an image has multiple classes, we prioritize animal > person > vehicle
                sorted_detections = sorted(valid_detections, 
                                          key=lambda x: (x.get('category', x.get('category_id', 99)), -x.get('conf', 0)))
                
                best_detection = sorted_detections[0]
                # Get category_id, accounting for different key names
                category_id = best_detection.get('category', best_detection.get('category_id', 99))
                
                # Get the appropriate folder based on class
                target_folder = class_folders.get(category_id, empty_dir)
        
        # Copy the image to the appropriate folder
        try:
            shutil.copy2(img_path, os.path.join(target_folder, img_filename))
            print(f"Copied {img_filename} to {os.path.basename(target_folder)} folder")
        except Exception as e:
            print(f"Error copying {img_path}: {e}")
    
    return animal_dir

def save_crops_by_class(detection_results, output_dir):
    """
    Save crops of detected objects by class category
    """
    # Create class-specific crop folders
    animal_crops_dir = os.path.join(output_dir, 'animal', 'crops')
    person_crops_dir = os.path.join(output_dir, 'person', 'crops')
    vehicle_crops_dir = os.path.join(output_dir, 'vehicle', 'crops')
    
    for folder in [animal_crops_dir, person_crops_dir, vehicle_crops_dir]:
        os.makedirs(folder, exist_ok=True)
    
    # Map class IDs to folders
    class_crop_folders = {
        0: animal_crops_dir,
        1: person_crops_dir,
        2: vehicle_crops_dir
    }
    
    # Process each image and save crops by class
    for img_result in detection_results:
        if 'detections' not in img_result or not img_result['detections']:
            continue
            
        # Extract the image path
        if 'image_id' in img_result:
            img_path = img_result['image_id']
        elif 'filename' in img_result:
            img_path = img_result['filename']
        else:
            print(f"Warning: Could not determine file path for result: {img_result}")
            continue
        
        img_filename = os.path.basename(img_path)
        
        try:
            # Load the image
            image = Image.open(img_path)
            img_width, img_height = image.size
            
            # Process each detection
            for i, detection in enumerate(img_result['detections']):
                # Get confidence with proper key fallback
                conf = detection.get('conf', detection.get('confidence', 0))
                if conf < DETECTION_THRESHOLD:
                    continue
                    
                # Get detection class and coordinates with proper key fallback
                category_id = detection.get('category', detection.get('category_id', 99))
                
                # Get bbox with proper key names
                if 'bbox' in detection:
                    bbox = detection['bbox']  # [x, y, width, height] normalized
                elif all(k in detection for k in ['x', 'y', 'width', 'height']):
                    bbox = [detection['x'], detection['y'], detection['width'], detection['height']]
                else:
                    print(f"Warning: Could not determine bbox for detection: {detection}")
                    continue
                
                # Convert normalized coordinates to pixel coordinates
                x, y, w, h = bbox
                x1 = int(x * img_width)
                y1 = int(y * img_height)
                x2 = int((x + w) * img_width)
                y2 = int((y + h) * img_height)
                
                # Crop the image
                crop = image.crop((x1, y1, x2, y2))
                
                # Determine destination folder based on class
                crop_folder = class_crop_folders.get(category_id)
                if crop_folder:
                    # Create crop filename
                    base_name = os.path.splitext(img_filename)[0]
                    crop_filename = f"{base_name}_crop_{i}.jpg"
                    crop_path = os.path.join(crop_folder, crop_filename)
                    
                    # Save the crop
                    crop.save(crop_path, "JPEG")
                    
        except Exception as e:
            print(f"Error processing crops for {img_path}: {e}")
            import traceback
            traceback.print_exc()
    
    return animal_crops_dir

def main():
    parser = argparse.ArgumentParser(description='Detect animals and classify crops')
    parser.add_argument('image_folder', help='Path to folder with images')
    args = parser.parse_args()
    
    # Create output directory inside the image folder
    image_folder = args.image_folder
    output_dir = os.path.join(image_folder, f'foxdetect_output_{MODEL_NAME}')
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device for detection model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Step 1: Run MegaDetector on images
    print("Running MegaDetector on images...")
    detection_model = pw_detection.MegaDetectorV6(device=device, pretrained=True, version="MDV6-yolov10-c")
    
    # Run batch detection
    detection_results = detection_model.batch_image_detection(
        image_folder, 
        batch_size=BATCH_SIZE
    )
    
    # Print a sample detection for debugging
    if detection_results and len(detection_results) > 0:
        print("\nSample detection result structure:")
        sample = detection_results[0]
        for key in sample:
            print(f"Key: {key}, Type: {type(sample[key])}")
        
        # If detections is a Supervision Detections object, handle differently
        if 'detections' in sample and hasattr(sample['detections'], 'xyxy'):
            print("\nUsing Supervision Detections format")
            # Convert to standard format for our functions
            converted_results = []
            for result in detection_results:
                img_path = result.get('img_id', '')
                detections_obj = result.get('detections', None)
                
                # Create a new result object with standard format
                new_result = {
                    'image_id': img_path,
                    'detections': []
                }
                
                # Check if we have valid detections
                if detections_obj is not None and len(detections_obj.xyxy) > 0:
                    # Get confidence scores
                    confidences = detections_obj.confidence if hasattr(detections_obj, 'confidence') else []
                    # Get class ids
                    class_ids = detections_obj.class_id if hasattr(detections_obj, 'class_id') else []
                    
                    # Iterate through each detection
                    for i in range(len(detections_obj.xyxy)):
                        x1, y1, x2, y2 = detections_obj.xyxy[i]
                        
                        # Get image dimensions to normalize coordinates
                        try:
                            img = Image.open(img_path)
                            img_width, img_height = img.size
                            img.close()
                        except Exception as e:
                            print(f"Warning: Could not open image {img_path} to get dimensions: {e}")
                            # Use placeholder values
                            img_width, img_height = 1, 1
                        
                        # Convert to normalized [x, y, width, height] format
                        x = float(x1) / img_width
                        y = float(y1) / img_height
                        width = float(x2 - x1) / img_width
                        height = float(y2 - y1) / img_height
                        
                        # Create detection entry
                        detection = {
                            'bbox': [x, y, width, height],
                            'conf': float(confidences[i]) if i < len(confidences) else 0.5,
                            'category_id': int(class_ids[i]) if i < len(class_ids) else 0  # Default to animal (0)
                        }
                        
                        new_result['detections'].append(detection)
                
                converted_results.append(new_result)
            
            # Replace original results with converted results
            detection_results = converted_results
    
    # Save detection results
    detection_json = os.path.join(output_dir, 'detections.json')
    with open(detection_json, 'w') as f:
        json.dump(detection_results, f, indent=2)
    print(f"Saved detection results to {detection_json}")
    
    # Draw detection boxes on images
    print("Drawing detection boxes...")
    detection_img_dir = os.path.join(output_dir, 'detection_images')
    os.makedirs(detection_img_dir, exist_ok=True)
    
    # Draw bounding boxes manually
    for result in detection_results:
        img_path = result.get('image_id', '')
        if not img_path or not os.path.exists(img_path):
            print(f"Warning: Image path not found: {img_path}")
            continue
            
        try:
            # Load image
            img = Image.open(img_path)
            draw = ImageDraw.Draw(img)
            
            # Draw each detection
            for detection in result.get('detections', []):
                bbox = detection.get('bbox', [0, 0, 0, 0])
                conf = detection.get('conf', 0)
                category_id = detection.get('category_id', 99)
                
                if conf >= DETECTION_THRESHOLD:
                    # Convert normalized coordinates to pixel values
                    img_width, img_height = img.size
                    x, y, w, h = bbox
                    x1 = int(x * img_width)
                    y1 = int(y * img_height)
                    x2 = int((x + w) * img_width)
                    y2 = int((y + h) * img_height)
                    
                    # Set color based on category
                    if category_id == 0:  # Animal
                        color = "green"
                    elif category_id == 1:  # Person
                        color = "blue"
                    elif category_id == 2:  # Vehicle
                        color = "orange"
                    else:
                        color = "red"
                        
                    # Draw bounding box
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                    
                    # Draw label
                    label = f"Class {category_id}: {conf:.2f}"
                    draw.text((x1, y1-15), label, fill=color)
            
            # Save the annotated image
            output_filename = os.path.join(detection_img_dir, os.path.basename(img_path))
            img.save(output_filename)
            
        except Exception as e:
            print(f"Error drawing detection for {img_path}: {e}")
    
    # Step 3: Organize images by MegaDetector class
    print("Organizing images by MegaDetector class...")
    animal_dir = organize_by_megadetector_class(detection_results, output_dir)
    
    # Step 4: Save crops by class
    print("Saving crops by MegaDetector class...")
    animal_crops_dir = save_crops_by_class(detection_results, output_dir)
    
    # Step 5: Run classification only on animal crops
    print("Loading classification model...")
    classification_model = load_classification_model()
    
    if classification_model and os.path.exists(animal_crops_dir) and os.listdir(animal_crops_dir):
        print(f"Classifying animal crops from {animal_crops_dir}...")
        classification_csv = os.path.join(output_dir, 'animal_classification_results.csv')
        classify_images(
            crop_dir=animal_crops_dir,
            model=classification_model,
            output_csv=classification_csv
        )
        print(f"Classification complete. Results saved to {classification_csv}")
        
        # Step 6: Plot classified animal images in a grid
        print("Creating grid visualization for animal classifications...")
        try:
            results_df = pd.read_csv(classification_csv)
            
            # Use the exact column names from your CSV
            image_col = 'crop_file'  # Your actual column name for the image path
            class_col = 'class'      # Your actual column name for the class/label
            conf_col = 'confidence'  # Your actual column name for the confidence score
            
            # Get up to 16 crops for visualization
            crop_files = list(results_df[image_col])[:16]
            labels = list(results_df[class_col])[:16]
            confidences = list(results_df[conf_col])[:16]
            
            # Create a grid
            num_images = len(crop_files)
            grid_size = min(4, int(np.ceil(np.sqrt(num_images))))
            
            fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
            axes = axes.flatten()
            
            # Plot each image with its label and confidence
            for i, (crop_file, label, confidence) in enumerate(zip(crop_files, labels, confidences)):
                if i >= len(axes):
                    break
                
                # Get full path to the image
                full_path = os.path.join(animal_crops_dir, crop_file)
                
                try:
                    img = np.array(Image.open(full_path))
                    axes[i].imshow(img)
                    axes[i].set_title(f"{label}\n{confidence:.2f}", fontsize=10)
                    axes[i].axis('off')
                except Exception as e:
                    print(f"Error loading image {full_path}: {e}")
                    axes[i].text(0.5, 0.5, f"Error loading\n{os.path.basename(crop_file)}", 
                                ha='center', va='center')
                    axes[i].axis('off')
            
            # Turn off any unused subplots
            for j in range(num_images, len(axes)):
                axes[j].axis('off')
            
            # Save the grid visualization
            plt.tight_layout()
            grid_output = os.path.join(output_dir, 'animal_classification_grid.png')
            plt.savefig(grid_output, dpi=200)
            plt.close(fig)
            print(f"Grid visualization saved to {grid_output}")
            
        except Exception as e:
            print(f"Error creating grid visualization: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("No animal crops found or classification model not loaded. Skipping classification.")
    
    # Step 5: Run classification only on animal crops
    print("Loading classification model...")
    classification_model = load_classification_model()
    
    if classification_model and os.path.exists(animal_crops_dir) and os.listdir(animal_crops_dir):
        print(f"Classifying animal crops from {animal_crops_dir}...")
        classification_csv = os.path.join(output_dir, 'animal_classification_results.csv')
        classify_images(
            crop_dir=animal_crops_dir,
            model=classification_model,
            output_csv=classification_csv
        )
        print(f"Classification complete. Results saved to {classification_csv}")
        
        # Step 6: Plot classified animal images in a grid
        print("Creating grid visualization for animal classifications...")
        try:
            results_df = pd.read_csv(classification_csv)
            
            # Use the exact column names from your CSV
            image_col = 'crop_file'  # Your actual column name for the image path
            class_col = 'class'      # Your actual column name for the class/label
            conf_col = 'confidence'  # Your actual column name for the confidence score
            
            # Get up to 16 crops for visualization
            crop_files = list(results_df[image_col])[:16]
            labels = list(results_df[class_col])[:16]
            confidences = list(results_df[conf_col])[:16]
            
            # Create a grid
            num_images = len(crop_files)
            grid_size = min(4, int(np.ceil(np.sqrt(num_images))))
            
            fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
            axes = axes.flatten()
            
            # Plot each image with its label and confidence
            for i, (crop_file, label, confidence) in enumerate(zip(crop_files, labels, confidences)):
                if i >= len(axes):
                    break
                
                # Get full path to the image
                full_path = os.path.join(animal_crops_dir, crop_file)
                
                try:
                    img = np.array(Image.open(full_path))
                    axes[i].imshow(img)
                    axes[i].set_title(f"{label}\n{confidence:.2f}", fontsize=10)
                    axes[i].axis('off')
                except Exception as e:
                    print(f"Error loading image {full_path}: {e}")
                    axes[i].text(0.5, 0.5, f"Error loading\n{os.path.basename(crop_file)}", 
                                ha='center', va='center')
                    axes[i].axis('off')
            
            # Turn off any unused subplots
            for j in range(num_images, len(axes)):
                axes[j].axis('off')
            
            # Save the grid visualization
            plt.tight_layout()
            grid_output = os.path.join(output_dir, 'animal_classification_grid.png')
            plt.savefig(grid_output, dpi=200)
            plt.close(fig)
            print(f"Grid visualization saved to {grid_output}")
            
        except Exception as e:
            print(f"Error creating grid visualization: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("No animal crops found or classification model not loaded. Skipping classification.")

if __name__ == "__main__":
    main()