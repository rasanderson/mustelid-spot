import os
import argparse
import csv
import numpy as np
import torch
import tensorflow as tf
from PIL import Image
from tqdm import tqdm
from PytorchWildlife.models import detection as pw_detection
import PytorchWildlife.utils as pw_utils

# Fixed paths and settings
MODEL_PATH = r"C:/Users/c0062193.CAMPUS/OneDrive - Newcastle University/General - Fox-AI/AI results/Model performance/fox_v7_20250305_115154/models/best_model_fox_v7_20250305_115154.h5"
CLASS_NAMES = ['fox', 'lagomorph', 'person', 'squirrel', 'badger', 'dog']
TARGET_SIZE = (224, 224)
DETECTION_THRESHOLD = 0.2
CLASSIFICATION_THRESHOLD = 0.7
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

def main():
    parser = argparse.ArgumentParser(description='Detect animals and classify crops')
    parser.add_argument('image_folder', help='Path to folder with images')
    args = parser.parse_args()
    
    # Create output directory inside the image folder
    image_folder = args.image_folder
    output_dir = os.path.join(image_folder, 'foxdetect_output')
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
    
    # Step 2: Save detection results
    detection_json = os.path.join(output_dir, 'detections.json')
    pw_utils.save_detection_json(
        detection_results, 
        detection_json,
        categories=detection_model.CLASS_NAMES,
        exclude_category_ids=[1, 2]  # Exclude person and vehicle, keep only animals (category 0)
    )
    
    # Draw detection boxes on images
    detection_img_dir = os.path.join(output_dir, 'detection_images')
    os.makedirs(detection_img_dir, exist_ok=True)
    pw_utils.save_detection_images(detection_results, detection_img_dir, overwrite=True)
    
    # Step 3: Crop the detected animals
    crops_dir = os.path.join(output_dir, 'crops')
    os.makedirs(crops_dir, exist_ok=True)
    pw_utils.save_crop_images(
        detection_results, 
        crops_dir, 
		overwrite=True
    )
    
    # Step 4: Run classification on crops
    print("Loading classification model...")
    classification_model = load_classification_model()
    
    if classification_model:
        print("Classifying cropped images...")
        classification_csv = os.path.join(output_dir, 'classification_results.csv')
        classify_images(
            crop_dir=crops_dir,
            model=classification_model,
            output_csv=classification_csv
        )
        print(f"Classification complete. Results saved to {classification_csv}")
    else:
        print("Classification model could not be loaded. Skipping classification step.")
    
    print(f"Process complete. Results saved to {output_dir}")

if __name__ == "__main__":
    main()