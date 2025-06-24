# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 11:38:00 2025

@author: nicho
"""
# Import require packages
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
import pandas as pd
from collections import defaultdict
import glob
import os

# Data directory paths
notfox_image_folder = "F:/Fox-AI/To process/Flux/newrabbit/"

# This will search subdirectories
notfox_files = glob.glob(f"{notfox_image_folder}**/*.png", recursive=True)

# Print results
print(f"Total non fox images: {len(notfox_files)}")

# Assign different classes
def classify_images(base_path, valid_classes):
    """
    Classify images based on their filenames instead of parent folder names.
    
    Args:
        base_path (str): Base path containing image files
        valid_classes (list): List of valid class names
    
    Returns:
        pd.DataFrame: DataFrame with image paths and their classifications
    """
    # Convert classes to lowercase for case-insensitive matching
    valid_classes = [cls.lower() for cls in valid_classes]
    
    # Initialize data structures with all required columns
    image_data = {
        'path': [],
        'filename': [],
        'class': [],
        'parent_folders': []
    }
    
    # Use the existing file lists
    for img_path in notfox_files:
        path = Path(img_path)
        filename = path.stem.lower()  # Get filename without extension and convert to lowercase
        parents = [p.name.lower() for p in path.parents]
        
        # Find all matching classes in filename
        assigned_classes = set()  # Using set to avoid duplicates
        for class_name in valid_classes:
            if class_name in filename:  # Check if class name appears in filename
                assigned_classes.add(class_name)
        
        # If image matches multiple classes, it will be counted in each
        if assigned_classes:
            for class_name in assigned_classes:
                image_data['path'].append(str(path))
                image_data['filename'].append(path.name)
                image_data['class'].append(class_name)
                image_data['parent_folders'].append('->'.join(reversed(parents)))
        else:
            # If no matches, still record the image but with no class
            image_data['path'].append(str(path))
            image_data['filename'].append(path.name)
            image_data['class'].append(None)
            image_data['parent_folders'].append('->'.join(reversed(parents)))
        
    # Create DataFrame
    df = pd.DataFrame(image_data)
    
    # Add a column for unclassified images
    df['is_classified'] = df['class'].notna()
    
    # Print summary
    print("\nClassification Summary:")
    print(f"Total images found: {len(df)}")
    print(f"Classified images: {df['is_classified'].sum()}")
    print(f"Unclassified images: {(~df['is_classified']).sum()}")
    print("\nClass distribution:")
    print(df['class'].value_counts(dropna=False))
    
    return df

def process_crop_images(df, class_name, target_size=(224, 224), channels=1, output_base_dir='F:/Fox-AI/Processed/FLUX'):
    """
    Process cropped images of a specific class and convert to numpy array.
    
    Args:
        df (pd.DataFrame): DataFrame containing image data
        class_name (str): Class to process
        target_size (tuple): Target size for resizing (x, y)
        channels (int): Number of channels (1 for grayscale, 3 for RGB)
        output_base_dir (str): Base directory for output
    
    Returns:
        tuple: (numpy array of processed images, list of processed filenames)
    """
    # Filter DataFrame for specified class and 'crop' in filename
    # Exclude None/empty classes
    # First ensure filename column contains only strings
    df['filename'] = df['filename'].astype(str)
    
    mask = (
        (df['class'] == class_name) & 
        (df['filename'].str.contains('crop', case=False, na=False)) &
        (df['class'].notna())
    )
    filtered_df = df[mask]
    
    print(f"Found {len(filtered_df)} cropped images for class '{class_name}'")
    
    # Initialize list to store processed images
    processed_images = []
    processed_filenames = []
    
    # Create output directory for this class
    output_folder = f"{output_base_dir}/{class_name}"
    Path(output_folder).mkdir(exist_ok=True, parents=True)
    
    for idx, row in filtered_df.iterrows():
        try:
            # Open image
            img = Image.open(row['path'])
            
            # Convert to grayscale if channels=1
            if channels == 1:
                img = img.convert('L')
            elif channels == 3:
                img = img.convert('RGB')
            
            # Resize
            img = img.resize(target_size, Image.Resampling.LANCZOS)
            
            # Convert to numpy array
            img_array = np.array(img)
            
            # Reshape if needed
            if channels == 1:
                img_array = img_array.reshape((*target_size, 1))
            
            # Normalize to [0, 1]
            img_array = img_array.astype(np.float32) / 255.0
            
            # Convert back to 0-255 range and correct data type for saving
            save_img = (img_array * 255).astype(np.uint8)
            if channels == 1:
                save_img = save_img.squeeze()  # Remove single channel dimension if grayscale
            
            # Save using cv2
            output_path = str(Path(output_folder) / row['filename'])
            cv2.imwrite(output_path, save_img)
                        
            # Add to lists
            processed_images.append(img_array)
            processed_filenames.append(row['filename'])
            
        except Exception as e:
            print(f"Error processing {row['filename']}: {str(e)}")
            continue
    
    if not processed_images:
        print(f"Warning: No valid images found for class '{class_name}'")
        return None, []
    
    # Stack all images into single array
    images_array = np.stack(processed_images)
    
    print(f"Processed {len(processed_images)} images for {class_name}")
    print(f"Array shape: {images_array.shape}")
    
    return images_array, processed_filenames

# Main execution
if __name__ == "__main__":
    # Define all animal classes to process
    animal_classes = ['badger', 'bird', 'boar', 'deer', 'dog', 'human', 'lagomorph', 'squirrel', 'hare'] #'rabbit', 
    
    # Additional valid classes that might be in the data but not in our processing list
    all_classes = animal_classes + ['person', 'empty', 'cat', 'animal', 'hare']
    
    # Set parameters
    target_size = (224, 224)
    channels = 1  # 1 for grayscale, 3 for RGB
    output_base_dir = 'F:/Fox-AI/Processed/FLUX/newlago'
    
    # Classify all images first
    df = classify_images(notfox_image_folder, all_classes)
    
    # Create a dictionary to store results for each class
    results = {}
    
    # Process each animal class
    for class_name in animal_classes:
        print(f"\n{'='*50}")
        print(f"Processing {class_name} images...")
        print(f"{'='*50}")
        
        try:
            images, filenames = process_crop_images(df, class_name, target_size, channels, output_base_dir)
            
            if images is not None:
                results[class_name] = {
                    'images': images,
                    'filenames': filenames,
                    'count': len(filenames)
                }
                
                print(f"\nExample filenames for {class_name}:")
                print(filenames[:min(5, len(filenames))])
            
        except Exception as e:
            print(f"Error processing class {class_name}: {str(e)}")
    
    # Print summary of all processed classes
    print("\n\nProcessing Summary:")
    print("="*50)
    for class_name, data in results.items():
        print(f"{class_name}: {data['count']} images processed")
    
    print("\nAll processing complete!")