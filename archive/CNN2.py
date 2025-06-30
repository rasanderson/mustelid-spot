import os
import sys
import json
import random
import subprocess
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from PIL import Image
import tensorflow as tf
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from collections import Counter
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import set_random_seed
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Resizing, Lambda
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_recall_curve, average_precision_score 

def setup_gpu():
    """Check for GPU availability and configure TensorFlow accordingly."""
    try:
        # Check if GPU is available
        physical_devices = tf.config.list_physical_devices('GPU')
        
        if physical_devices:
            print("GPU(s) detected:")
            for device in physical_devices:
                print(f"  {device}")
            
            # Configure TensorFlow to use GPU memory growth
            for gpu in physical_devices:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            print("GPU memory growth enabled")
            return True
        else:
            print("No GPU detected. Running on CPU.")
            # Set TensorFlow to use CPU only
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            return False
            
    except Exception as e:
        print(f"Error setting up GPU: {str(e)}")
        print("Defaulting to CPU.")
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        return False

def create_output_directory(base_output_dir, run_name):
    """Create a named output directory to save results."""
    # Use the run name as part of the output directory
    output_dir = os.path.join(base_output_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create subdirectories
    plots_dir = os.path.join(output_dir, "plots")
    models_dir = os.path.join(output_dir, "models")
    
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    print(f"Created output directory: {output_dir}")
    return output_dir, plots_dir, models_dir

def get_image_time(img_path):
    """
    Extract timestamp from image metadata or filename
    Returns datetime object
    """
    try:
        # First try to get time from image EXIF data
        img = Image.open(img_path)
        if hasattr(img, '_getexif') and img._getexif():
            exif = {
                PIL.ExifTags.TAGS[k]: v
                for k, v in img._getexif().items()
                if k in PIL.ExifTags.TAGS
            }
            if 'DateTimeOriginal' in exif:
                return datetime.strptime(exif['DateTimeOriginal'], '%Y:%m:%d %H:%M:%S')
    except Exception:
        pass
    
    # Fall back to file creation/modification time if EXIF fails
    try:
        file_time = os.path.getmtime(img_path)
        return datetime.fromtimestamp(file_time)
    except Exception:
        # If all else fails, return current time (this is a fallback that should rarely be used)
        return datetime.now()

def group_images_into_sequences(image_paths, time_threshold_seconds=30):
    """
    Group images into sequences based on their capture time
    Returns a dictionary mapping sequence IDs to lists of image paths
    """
    if not image_paths:
        return {}
    
    # Get timestamps for all images
    image_times = []
    for path in tqdm(image_paths, desc="Reading image timestamps"):
        image_times.append((path, get_image_time(path)))
    
    # Sort by timestamp
    image_times.sort(key=lambda x: x[1])
    
    # Group into sequences
    sequences = {}
    current_seq_id = 0
    current_seq = []
    last_time = None
    
    for path, time in image_times:
        if last_time is None or (time - last_time).total_seconds() <= time_threshold_seconds:
            # Add to current sequence
            current_seq.append(path)
        else:
            # Start a new sequence
            if current_seq:
                sequences[current_seq_id] = current_seq
                current_seq_id += 1
                current_seq = [path]
        
        last_time = time
    
    # Add the last sequence if it has images
    if current_seq:
        sequences[current_seq_id] = current_seq
    
    return sequences

def build_sequence_cache(base_dir, classes, cache_file, time_threshold_seconds=30):
    """
    Build a cache file containing sequence assignments for all images
    """
    print(f"Building sequence cache for dataset in {base_dir}")
    
    # Dictionary to store all image paths by class
    all_images_by_class = {cls_idx: [] for cls_idx in range(len(classes))}
    
    # Load fox images (class 0)
    fox_dir = Path(base_dir) / "Fox"
    print(f"\nFinding fox images from {fox_dir}")
    
    for img_path in fox_dir.glob("**/*.jp*g"):
        # Only include images with 'crop' in the filename
        if 'crop' in img_path.name.lower():
            all_images_by_class[0].append(str(img_path))
    
    print(f"Found {len(all_images_by_class[0])} fox images")
    
    # Load non-fox images (classes 1+)
    notfox_dir = Path(base_dir) / "Not Fox/preprocessed"
    print(f"\nFinding non-fox images from {notfox_dir}")
    
    for folder in notfox_dir.iterdir():
        if not folder.is_dir():
            continue
        
        try:
            # Extract the class name from the folder name
            folder_name = folder.name.lower()
            
            # Find which class this folder corresponds to
            for class_idx, class_name in enumerate(classes):
                if class_idx > 0 and class_name in folder_name:
                    print(f"\nProcessing {class_name} images from {folder}")
                    
                    for img_path in folder.glob("**/*.jp*g"):
                        all_images_by_class[class_idx].append(str(img_path))
                    
                    print(f"Found {len(all_images_by_class[class_idx])} {class_name} images")
                    break
        except Exception as e:
            print(f"Error processing folder {folder}: {e}")
    
    # Dictionary to store image metadata with timestamps and sequence assignments
    image_metadata = {}
    
    # Process each class
    for cls_idx in range(len(classes)):
        class_name = classes[cls_idx] if cls_idx > 0 else 'fox'
        print(f"\nProcessing timestamps for {class_name} images...")
        
        # Get image timestamps
        image_times = []
        for path in tqdm(all_images_by_class[cls_idx], desc=f"Reading {class_name} timestamps"):
            timestamp = get_image_time(path)
            image_times.append((path, timestamp))
            # Store in metadata dictionary
            image_metadata[path] = {
                'class_idx': cls_idx,
                'timestamp': timestamp.isoformat(),  # Convert to string for JSON serialization
                'sequence_id': None  # Will be assigned in the next step
            }
        
        # Sort by timestamp
        image_times.sort(key=lambda x: x[1])
        
        # Assign sequence IDs
        current_seq_id = f"{cls_idx}_0"  # Prefix with class index for uniqueness
        last_time = None
        
        for path, time in image_times:
            if last_time is None or (time - last_time).total_seconds() <= time_threshold_seconds:
                # Add to current sequence
                image_metadata[path]['sequence_id'] = current_seq_id
            else:
                # Start a new sequence
                seq_num = int(current_seq_id.split('_')[1]) + 1
                current_seq_id = f"{cls_idx}_{seq_num}"
                image_metadata[path]['sequence_id'] = current_seq_id
            
            last_time = time
    
    # Count sequences for reporting
    sequences_by_class = {}
    for cls_idx in range(len(classes)):
        sequences_by_class[cls_idx] = set()
        for path, metadata in image_metadata.items():
            if metadata['class_idx'] == cls_idx:
                sequences_by_class[cls_idx].add(metadata['sequence_id'])
    
    # Print sequence statistics
    print("\nSequence grouping statistics:")
    for cls_idx in range(len(classes)):
        cls_name = classes[cls_idx] if cls_idx > 0 else 'fox'
        seq_count = len(sequences_by_class[cls_idx])
        img_count = len([p for p, m in image_metadata.items() if m['class_idx'] == cls_idx])
        print(f"{cls_name}: {img_count} images in {seq_count} sequences")
    
    # Save metadata to cache file
    print(f"\nSaving sequence cache to {cache_file}")
    with open(cache_file, 'w') as f:
        json.dump(image_metadata, f)
    
    print("Sequence cache created successfully!")
    return image_metadata

def load_sequence_cache(cache_file):
    """
    Load the sequence cache from file
    """
    print(f"Loading sequence cache from {cache_file}")
    with open(cache_file, 'r') as f:
        image_metadata = json.load(f)
    
    return image_metadata

def split_by_sequences_from_cache(image_metadata, classes, val_size=0.15, test_size=0.15, random_state=42):
    """
    Splits the dataset into train, validation, and test sets based on cached sequence IDs
    Returns dictionaries for each split containing images by class
    """
    train_size = 1 - val_size - test_size
    
    # Get all sequence IDs by class
    sequences_by_class = {}
    for cls_idx in range(len(classes)):
        sequences_by_class[cls_idx] = set()
        for path, metadata in image_metadata.items():
            if metadata['class_idx'] == cls_idx:
                sequences_by_class[cls_idx].add(metadata['sequence_id'])
        
        # Convert to list for splitting
        sequences_by_class[cls_idx] = list(sequences_by_class[cls_idx])
    
    # Split sequences for each class
    train_images_by_class = {cls_idx: [] for cls_idx in range(len(classes))}
    val_images_by_class = {cls_idx: [] for cls_idx in range(len(classes))}
    test_images_by_class = {cls_idx: [] for cls_idx in range(len(classes))}
    
    for cls_idx in range(len(classes)):
        if not sequences_by_class[cls_idx]:
            continue
            
        # Get sequence IDs for this class
        seq_ids = sequences_by_class[cls_idx]
        
        # Split sequence IDs into train and temp (temporary holding for val and test)
        train_seq_ids, temp_seq_ids = train_test_split(
            seq_ids, test_size=(val_size + test_size), random_state=random_state
        )
        
        # Split temp into val and test
        temp_val_size = val_size / (val_size + test_size)
        val_seq_ids, test_seq_ids = train_test_split(
            temp_seq_ids, test_size=(1-temp_val_size), random_state=random_state
        )
        
        # Assign all images from each sequence to the appropriate split
        for path, metadata in image_metadata.items():
            if metadata['class_idx'] == cls_idx:
                seq_id = metadata['sequence_id']
                
                if seq_id in train_seq_ids:
                    train_images_by_class[cls_idx].append(path)
                elif seq_id in val_seq_ids:
                    val_images_by_class[cls_idx].append(path)
                elif seq_id in test_seq_ids:
                    test_images_by_class[cls_idx].append(path)
    
    # Print statistics 
    print("\nSequence-based split statistics:")
    for cls_idx in range(len(classes)):
        cls_name = classes[cls_idx] if cls_idx > 0 else 'fox'
        print(f"\n{cls_name}:")
        print(f"  Training images: {len(train_images_by_class[cls_idx])}")
        print(f"  Validation images: {len(val_images_by_class[cls_idx])}")
        print(f"  Test images: {len(test_images_by_class[cls_idx])}")
    
    return train_images_by_class, val_images_by_class, test_images_by_class

def balance_dataset(images_by_class, min_count=None):
    """
    Balance the dataset by sampling the same number of images from each class
    If min_count is None, uses the smallest class size
    """
    # Find minimum count across classes if not specified
    if min_count is None:
        min_count = min([len(images) for images in images_by_class.values() if len(images) > 0])
    
    balanced_images_by_class = {}
    
    for cls_idx, images in images_by_class.items():
        if len(images) == 0:
            balanced_images_by_class[cls_idx] = []
            continue
            
        # Randomly sample min_count images
        balanced_images_by_class[cls_idx] = random.sample(images, min(min_count, len(images)))
    
    return balanced_images_by_class

def load_balanced_split_dataset_from_cache(cache_file, classes, val_size=0.15, test_size=0.15, 
                                          random_state=42, augment=True):
    """
    Load and split dataset based on cached sequences, then balance each split
    Returns arrays ready for model training
    """
    # Load sequence cache
    image_metadata = load_sequence_cache(cache_file)
    
    # Split dataset based on sequences
    train_by_class, val_by_class, test_by_class = split_by_sequences_from_cache(
        image_metadata, classes, val_size, test_size, random_state
    )
    
    # Balance each split separately
    train_by_class = balance_dataset(train_by_class)
    val_by_class = balance_dataset(val_by_class)
    test_by_class = balance_dataset(test_by_class)
    
    # Load and preprocess images for each split
    print("\nLoading and preprocessing images...")
    
    # For training set
    X_train, y_train = [], []
    for cls_idx, images in train_by_class.items():
        for img_path in tqdm(images, desc=f"Loading training images for class {cls_idx}"):
            result = load_and_preprocess_image(img_path, augment=augment)
            if augment and isinstance(result, list):
                for img in result:
                    X_train.append(img)
                    y_train.append(cls_idx)
            else:
                X_train.append(result)
                y_train.append(cls_idx)
    
    # For validation set
    X_val, y_val = [], []
    for cls_idx, images in val_by_class.items():
        for img_path in tqdm(images, desc=f"Loading validation images for class {cls_idx}"):
            result = load_and_preprocess_image(img_path, augment=False)  # No augmentation for validation
            X_val.append(result)
            y_val.append(cls_idx)
    
    # For test set
    X_test, y_test = [], []
    for cls_idx, images in test_by_class.items():
        for img_path in tqdm(images, desc=f"Loading test images for class {cls_idx}"):
            result = load_and_preprocess_image(img_path, augment=False)  # No augmentation for test
            X_test.append(result)
            y_test.append(cls_idx)
    
    # Convert to numpy arrays
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_val = np.array(X_val)
    y_val = np.array(y_val)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    # Create class counts for reporting
    class_counts = {}
    for i, cls in enumerate(classes):
        cls_name = cls if i > 0 else 'fox'
        train_count = len(train_by_class[i])
        val_count = len(val_by_class[i])
        test_count = len(test_by_class[i])
        class_counts[cls_name] = {
            'train': train_count * (2 if augment else 1),  # Account for augmentation
            'val': val_count,
            'test': test_count,
            'total': train_count * (2 if augment else 1) + val_count + test_count
        }
    
    print("\nDataset splitting and loading complete!")
    print(f"Training set: {X_train.shape[0]} images")
    print(f"Validation set: {X_val.shape[0]} images")
    print(f"Test set: {X_test.shape[0]} images")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, class_counts

def train_model_with_cached_seq_split(base_dir, classes, output_dir, run_name, 
                                     cache_file=None, rebuild_cache=False,
                                     val_size=0.15, test_size=0.15, random_state=42,
                                     epochs=100, batch_size=32):
    """
    Main function to train the model with cached sequence-based splitting.
    
    Parameters:
    - base_dir: Directory containing the images
    - classes: List of class names
    - output_dir: Directory to save outputs
    - run_name: Name for this training run
    - cache_file: Path to sequence cache file (will be created if it doesn't exist)
    - rebuild_cache: Force rebuilding the cache even if it exists
    - val_size, test_size: Validation and test set proportions
    - random_state: For reproducibility
    - epochs, batch_size: Training parameters
    """
    import os
    import numpy as np
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    
    print(f"Starting training process with {len(classes)} classes")
    print(f"Results will be saved to: {output_dir}")
    
    # Create output directories
    output_dir, plots_dir, models_dir = create_output_directory(output_dir, run_name)
    
    # Default cache file if not specified
    if cache_file is None:
        cache_file = os.path.join(output_dir, "sequence_cache.json")
    
    # Create or load the sequence cache
    if rebuild_cache or not os.path.exists(cache_file):
        image_metadata = build_sequence_cache(base_dir, classes, cache_file)
    
    # Split and load the dataset based on cached sequences
    X_train, y_train, X_val, y_val, X_test, y_test, class_counts = load_balanced_split_dataset_from_cache(
        cache_file, classes, val_size=val_size, test_size=test_size, random_state=random_state
    )
    
    print(f"\nTotal dataset size: {len(X_train) + len(X_val) + len(X_test)} images")
    print(f"Training set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    print(f"Test set size: {len(X_test)}")
    
    # Find which classes are actually present in the dataset
    all_labels = np.concatenate([y_train, y_val, y_test])
    unique_classes = np.unique(all_labels)
    present_classes = [classes[i] for i in unique_classes]
    
    print("\nClasses present in dataset:")
    for i, class_idx in enumerate(unique_classes):
        count = np.sum(all_labels == class_idx)
        print(f"{classes[class_idx]}: {count} images (class index {class_idx})")
    
    print("\nMissing classes:")
    missing_classes = [cls for i, cls in enumerate(classes) if i not in unique_classes]
    for cls in missing_classes:
        print(f"- {cls}")
    
    # Create and train model using only the number of classes actually present
    num_classes = len(unique_classes)
    print(f"\nCreating model with {num_classes} output classes")
    model = create_model(num_classes)
    
    # Define Early Stopping and Model Checkpoint callbacks
    early_stopping = EarlyStopping(
        monitor='val_accuracy', patience=10, mode='max', 
        restore_best_weights=True, verbose=1
    )
    checkpoint = ModelCheckpoint(
        os.path.join(models_dir, f'best_model_{run_name}.h5'), 
        monitor='val_accuracy', 
        save_best_only=True
    )
    
    print("\nStarting model training...")
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, checkpoint],
        verbose=1
    )
    
    print("\nTraining complete! Generating evaluation plots...")
    
    # Plot and save training history
    plot_training_history(history, plots_dir, run_name)
    
    # Evaluate model and save results
    report = evaluate_model(model, X_test, y_test, present_classes, plots_dir, run_name)
    
    # Save model
    model_path = os.path.join(models_dir, f"fox_classifier_model_{run_name}.h5")
    model.save(model_path)
    print(f"Saved model to {model_path}")
    
    # Save training summary
    # Convert class_counts to format expected by save_training_summary
    summary_class_counts = {}
    for cls, counts in class_counts.items():
        summary_class_counts[cls] = counts['total']
        
    summary = save_training_summary(output_dir, present_classes, summary_class_counts, history, report, run_name)
    
    return model, history, summary

# Example usage in the main script:

if __name__ == "__main__":
    # Check if GPU is available
    using_gpu = setup_gpu()
    
    # Set batch size based on GPU availability
    if using_gpu:
        batch_size = 32  # Larger batch size for GPU
    else:
        batch_size = 4  # Smaller batch size for CPU
    
    # Data directory
    base_dir = "C:/Users/c0062193.CAMPUS/OneDrive - Newcastle University/General - Fox-AI/Processed/Cleaned/"
    
    # Output directory for saving results
    output_dir = "C:/Users/c0062193.CAMPUS/OneDrive - Newcastle University/General - Fox-AI/AI results/Model performance/"

    # Set a name for this training run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"fox_v10_seq_split_{timestamp}"
    
    # Class definitions
    classes = ['fox', 'lagomorph', 'person', 'squirrel', 'badger', 'dog', 'bird', 'deer', 'muntjack', 'boar']
    
    # Cache file path
    cache_file = os.path.join(output_dir, "sequence_cache.json")
    
    # Train model with cached sequence-based splitting
    model, history, summary = train_model_with_cached_seq_split(
        base_dir, 
        classes, 
        output_dir, 
        run_name=run_name,
        cache_file=cache_file,
        rebuild_cache=False,  # Set to True to force rebuilding the cache
        val_size=0.15,
        test_size=0.15,
        random_state=42,
        epochs=100, 
        batch_size=batch_size
    )
    
    print("\nTraining and evaluation completed successfully!")
    print(f"All results saved to {output_dir}")
