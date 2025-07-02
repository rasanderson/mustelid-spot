"""
Image preprocessing pipeline for camera trap dataset classification.

Automatically classifies images based on folder structure and preprocesses them 
for training. Searches through directory hierarchies to find images, assigns 
class labels based on parent folder names, and converts to standardized arrays.

Key Functions:
- classify_images(): Auto-assigns classes based on folder names (supports multi-class)
- process_crop_images(): Converts images to grayscale/RGB arrays with normalization
- Filters for cropped images (filename contains 'crop')
- Saves preprocessed images to organized output directories

Supported Classes: person, fox, bird, dog, lagomorph, deer, squirrel, badger, 
empty, cat, animal, boar, human, rabbit, hare

Output: Normalized numpy arrays (224x224) ready for model training, plus 
organized preprocessed image files saved to disk.

Usage: Update folder paths, specify target class, run to get training-ready arrays
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

# home
#fox_image_folder = 'C:/Users/nicho/OneDrive - Newcastle University/General - Fox-AI/Processed/Fox/'
#notfox_image_folder = 'C:/Users/nicho/OneDrive - Newcastle University/General - Fox-AI/Processed/Not Fox/'

#uni
#fox_image_folder = 'C:/Users/c0062193.CAMPUS/OneDrive - Newcastle University/General - Fox-AI/Processed/Fox/'
notfox_image_folder = "G:/Data/data_v15/Lila/rabbit/crops" #'C:/Users/c0062193.CAMPUS/OneDrive - Newcastle University/General - Fox-AI/Processed/Not Fox/'

# This will search subdirectories
#fox_files = glob.glob(f"{fox_image_folder}**/*.jp*g", recursive=True)
notfox_files = glob.glob(f"{notfox_image_folder}**/*.jp*g", recursive=True)

# Print results
#print(f"Total fox images: {len(fox_files)}")
print(f"Total non fox images: {len(notfox_files)}")

# Assign different classes
def classify_images(base_path, valid_classes):
    """
    Classify images based on their immediate parent folder names.
    
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
        'parent_folders': [],
        'immediate_folder': []
    }
    
    # Use the existing file lists
    for img_path in notfox_files:
        path = Path(img_path)
        filename = path.name
        parents = [p.name.lower() for p in path.parents]
        
        # Get immediate parent folder name (the folder containing the image)
        immediate_folder = path.parent.name.lower()
        
        # Find all matching classes in immediate folder name
        assigned_classes = set()  # Using set to avoid duplicates
        for class_name in valid_classes:
            if class_name in immediate_folder:  # Check if class name appears in folder name
                assigned_classes.add(class_name)
        
        # If folder matches multiple classes, it will be counted in each
        if assigned_classes:
            for class_name in assigned_classes:
                image_data['path'].append(str(path))
                image_data['filename'].append(filename)
                image_data['class'].append(class_name)
                image_data['parent_folders'].append('->'.join(reversed(parents)))
                image_data['immediate_folder'].append(path.parent.name)
        else:
            # If no matches, still record the image but with no class
            image_data['path'].append(str(path))
            image_data['filename'].append(filename)
            image_data['class'].append(None)
            image_data['parent_folders'].append('->'.join(reversed(parents)))
            image_data['immediate_folder'].append(path.parent.name)
        
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
    print("\nFolder distribution:")
    print(df['immediate_folder'].value_counts())
    
    return df

# Example usage
if __name__ == "__main__":
    # Define valid classes
    classes = ['person', 'fox', 'bird', 'dog', 'lagomorph', 'deer', 'squirrel', 'badger', 'empty', 'cat', 'animal', 'boar', 'human', 'rabbit', 'hare']
    
    # Classify images
    df = classify_images(notfox_image_folder, classes)

# Preprocess the images per class into the right array

def process_crop_images(df, class_name, target_size=(224, 224), channels=1):
    """
    Process cropped images of a specific class and convert to numpy array.
    
    Args:
        df (pd.DataFrame): DataFrame containing image data
        class_name (str): Class to process
        target_size (tuple): Target size for resizing (x, y)
        channels (int): Number of channels (1 for grayscale, 3 for RGB)
    
    Returns:
        tuple: (numpy array of processed images, list of processed filenames)
    """
    # Filter DataFrame for specified class and 'crop' in filename
    # Exclude None/empty classes
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
            
            # Add after img_array is created but before appending to processed_images:
            #output_folder = 'C:/Users/c0062193.CAMPUS/OneDrive - Newcastle University/General - Fox-AI/Processed/Not Fox/' + f"preprocessed/preprocessed_{class_name}_images"
            output_folder = "G:/Data/data_v15/Lila/lagomorph/preprocessed_" + f"{class_name}_images"
            Path(output_folder).mkdir(exist_ok=True)
            
            # Convert back to 0-255 range and correct data type
            save_img = (img_array * 255).astype(np.uint8)
            if channels == 1:
                save_img = save_img.squeeze()  # Remove single channel dimension if grayscale
            
            # Save using cv2
            output_path = str(Path(output_folder) / row['filename'])
            cv2.imwrite(output_path, save_img)
                        
            # Add to df
            processed_images.append(img_array)
            processed_filenames.append(row['filename'])
            
        except Exception as e:
            print(f"Error processing {row['filename']}: {str(e)}")
            continue
    
    if not processed_images:
        raise ValueError(f"No valid images found for class '{class_name}'")
    
    # Stack all images into single array
    images_array = np.stack(processed_images)
    
    print(f"Processed {len(processed_images)} images")
    print(f"Array shape: {images_array.shape}")
    
    return images_array, processed_filenames

# Example usage
if __name__ == "__main__":
    # Assuming df is your DataFrame from the previous classification
    class_name = 'rabbit'  # choose your class
    target_size = (224, 224)  # specify desired size
    channels = 1  # 1 for grayscale, 3 for RGB
    
    try:
        images, filenames = process_crop_images(df, class_name, target_size, channels)
        print("\nExample filenames:")
        print(filenames[:5])
        
    except ValueError as e:
        print(f"Error: {str(e)}")
