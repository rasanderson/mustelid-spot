# CNN training with Claude's help for fox classification
# Nicholas Allen, SNES, Newcastle University

import os
import sys
import random
import subprocess
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
import tensorflow as tf
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from collections import Counter
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import set_random_seed
from sklearn.metrics import roc_curve, auc, f1_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Resizing, Lambda
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_recall_curve, average_precision_score 
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

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

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

def load_dataset(base_dir, classes, augment=True, val_split=0.1):
    """Load image paths and labels without loading actual images into memory."""
    # Dictionary to store image paths by class
    class_images = {cls_idx: [] for cls_idx in range(len(classes))}
    class_counts = {cls: 0 for cls in classes}
    
    print(f"Starting to load dataset from {base_dir}")
    
    # Load fox images (class 0)
    fox_dir = Path(base_dir) / "Fox"
    print(f"\nLoading fox images from {fox_dir}")
    fox_count = 0
    fox_patterns = list(fox_dir.glob("**/*.jpg")) + list(fox_dir.glob("**/*.jpeg")) + list(fox_dir.glob("**/*.JPG")) + list(fox_dir.glob("**/*.JPEG")) + list(fox_dir.glob("**/*.png")) + list(fox_dir.glob("**/*.PNG"))
    
    for img_path in fox_patterns:
        # Only include images with 'crop' in the filename
        if 'crop' in img_path.name.lower():
            try:
                # Store the path instead of loading the image now
                class_images[0].append(str(img_path))  # Convert Path to string
                fox_count += 1
                if fox_count % 100 == 0:
                    print(f"Found {fox_count} fox images")
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
    class_counts['fox'] = fox_count
    
    # Load non-fox images (classes 1+)
    notfox_dir = Path(base_dir) / "Not Fox/preprocessed/"
    print(f"\nFinding non-fox images from {notfox_dir}")
    
    for folder in notfox_dir.iterdir():
        if not folder.is_dir():
            continue
        
        try:
            # Extract the class name from the folder name
            folder_name = folder.name.lower()
            
            # Find which class this folder corresponds to
            class_found = False
            for class_idx, class_name in enumerate(classes):
                if class_idx > 0 and class_name in folder_name:
                    print(f"\nProcessing {class_name} images from {folder}")
                    class_count = 0
                    
                    img_patterns = list(folder.glob("**/*.jpg")) + list(folder.glob("**/*.jpeg")) + list(folder.glob("**/*.JPG")) + list(folder.glob("**/*.JPEG")) + list(folder.glob("**/*.png")) + list(folder.glob("**/*.PNG"))
                    for img_path in img_patterns:
                        try:
                            # Store the path instead of loading the image now
                            class_images[class_idx].append(str(img_path))  # Convert Path to string
                            class_count += 1
                            if class_count % 100 == 0:
                                print(f"Found {class_count} {class_name} images")
                        except Exception as e:
                            print(f"Error processing {img_path}: {e}")
                    
                    class_counts[class_name] = class_count
                    print(f"Finished finding {class_count} {class_name} images")
                    class_found = True
                    break
            
            if not class_found:
                print(f"Could not find matching class for folder {folder}")
                
        except Exception as e:
            print(f"Error processing folder {folder}: {e}")
    
    # Find the class with the minimum number of samples
    min_count = min([len(imgs) for imgs in class_images.values()])
    print(f"\nSmallest class has {min_count} images")
    
    # Randomly sample the same number of images from each class
    all_image_paths = []
    all_labels = []
    

    for class_idx, image_paths in class_images.items():
        # Randomly sample min_count images from this class
        sampled_paths = random.sample(image_paths, min_count)
        
        class_name = classes[class_idx] if class_idx > 0 else 'fox'
        print(f"Randomly sampled {min_count} images for class {class_name}")
        
        # Add paths and labels without loading images
        all_image_paths.extend(sampled_paths)
        all_labels.extend([class_idx] * len(sampled_paths))
    
    # Split into train and validation sets
    indices = list(range(len(all_image_paths)))
    random.shuffle(indices)
    split_idx = int(len(indices) * val_split)
    
    val_indices = indices[:split_idx]
    train_indices = indices[split_idx:]
    
    X_train = [all_image_paths[i] for i in train_indices]
    y_train = [all_labels[i] for i in train_indices]
    X_val = [all_image_paths[i] for i in val_indices]
    y_val = [all_labels[i] for i in val_indices]
    
    # Create final class counts dictionary 
    final_class_counts = {
        classes[i] if i > 0 else 'fox': min_count * (2 if augment else 1)
        for i in range(len(classes))
    }
    
    print(f"\nSplit dataset: {len(X_train)} training, {len(X_val)} validation images")
    
    return X_train, X_val, y_train, y_val, final_class_counts

def create_model(num_classes, input_shape=(224, 224, 1)):
    """Set CNN learning parameters"""
    # Initial learning rate
    initial_learning_rate = 0.001
    
    # Create an optimizer with a decreasing learning rate schedule
    lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate,
        decay_steps=1000,
        end_learning_rate=0.0001,
        power=1.0)  # Linear decay
    
    optimizer = Adam(learning_rate=lr_schedule)
	
    """Create and compile the CNN model."""
    model = Sequential([
	    # Input layer with the specified shape
        tf.keras.Input(shape=input_shape),
		
		# Image standardization
		Lambda(lambda x: tf.image.per_image_standardization(x), input_shape=input_shape, name='standardization'),

        # First block - 32 filters
        Conv2D(32, (3, 3), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        Conv2D(32, (3, 3), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        MaxPooling2D((2, 2)),
        BatchNormalization(),
        Dropout(0.2),
        
        # Second block - 64 filters
        Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        MaxPooling2D((2, 2)),
        BatchNormalization(),
        Dropout(0.2),
        
        # Third block - 128 filters
        Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        MaxPooling2D((2, 2)),
        BatchNormalization(),
        Dropout(0.2),
        
        # Fully connected layers
        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy', #binary_crossentropy #sparse_categorical_crossentropy
        metrics=['accuracy',tf.keras.metrics.SparseCategoricalAccuracy(), tf.keras.metrics.F1Score()]
    )
    
    print("\nModel architecture:")
    model.summary()
    return model

def plot_training_history(history, plots_dir, run_name):
    """Plot training and validation metrics and save the figure."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Accuracy plot
    ax1.plot(history.history['accuracy'], label='Training')
    ax1.plot(history.history['val_accuracy'], label='Validation')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # Loss plot
    ax2.plot(history.history['loss'], label='Training')
    ax2.plot(history.history['val_loss'], label='Validation')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    
    # Save the figure
    plot_path = os.path.join(plots_dir, f"training_history_{run_name}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved training history plot to {plot_path}")
    
    # Also save raw history data as CSV
    history_df = pd.DataFrame(history.history)
    history_df['run_name'] = run_name  # Use the same indentation style as the line above
    history_csv_path = os.path.join(plots_dir, f"training_history_{run_name}.csv")
    history_df.to_csv(history_csv_path, index=False)
    print(f"Saved training metrics to {history_csv_path}")
    
    plt.show()

def plot_confusion_matrix_with_histogram(y_true, y_pred, classes, plots_dir, run_name):
    """
    Plot confusion matrices alongside a histogram of class distribution.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    classes : list
        List of class names
    plots_dir : str
        Directory to save plots
    """
  
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create normalized confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Calculate class distribution
    class_counts = Counter(y_true)
    class_indices = sorted(class_counts.keys())
    counts = [class_counts[idx] for idx in class_indices]
    class_names = [classes[idx] for idx in class_indices]
    
    # Set up the figure with three subplots
    fig = plt.figure(figsize=(24, 8))
    
    # Define grid for subplots
    gs = fig.add_gridspec(1, 3, width_ratios=[4, 4, 2])
    
    # Plot raw counts
    ax1 = fig.add_subplot(gs[0])
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, ax=ax1)
    ax1.set_title('Raw Confusion Matrix', fontsize=14)
    ax1.set_ylabel('True Label', fontsize=12)
    ax1.set_xlabel('Predicted Label', fontsize=12)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    
    # Plot normalized values (as percentages)
    ax2 = fig.add_subplot(gs[1])
    sns.heatmap(cm_norm, annot=True, fmt='.1%', xticklabels=classes, yticklabels=classes, ax=ax2)
    ax2.set_title('Normalized Confusion Matrix', fontsize=14)
    ax2.set_ylabel('True Label', fontsize=12)
    ax2.set_xlabel('Predicted Label', fontsize=12)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
    
    # Plot class distribution histogram
    ax3 = fig.add_subplot(gs[2])
    bars = ax3.barh(class_names, counts, color='darkred')
    ax3.set_title('Class Distribution', fontsize=14)
    ax3.set_xlabel('Number of Samples', fontsize=12)
    ax3.set_ylabel('Class', fontsize=12)
    
    # Add count labels to bars
    for bar in bars:
        width = bar.get_width()
        ax3.text(width + width*0.05, 
                 bar.get_y() + bar.get_height()/2,
                 f'{width}', 
                 ha='left', 
                 va='center',
                 fontsize=10)
    
    plt.tight_layout()
    
    # Save the figure
    plot_path = os.path.join(plots_dir, f"confusion_matrix_{run_name}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved confusion matrix plot to {plot_path}")
    
    # Save confusion matrix as CSV
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    cm_csv_path = os.path.join(plots_dir, f"confusion_matrix_{run_name}.csv")
    cm_df.to_csv(cm_csv_path)
    
    # Save normalized confusion matrix as CSV
    cm_norm_df = pd.DataFrame(cm_norm, index=classes, columns=classes)
    cm_norm_csv_path = os.path.join(plots_dir, f"confusion_matrix_normalized_{run_name}.csv")
    cm_norm_df.to_csv(cm_norm_csv_path)
    
    print(f"Saved confusion matrices to CSV files")
    
    plt.show()

def plot_f1_scores(y_val, y_pred_classes, classes, plots_dir, run_name):
    """Plot F1 scores for each class and save the figure."""
    # Calculate F1 score for each class
    f1_scores = []
    for i in range(len(classes)):
        # Create binary classification for current class
        y_true_bin = (np.array(y_val) == i).astype(int)
        y_pred_bin = (np.array(y_pred_classes) == i).astype(int)
        f1 = f1_score(y_true_bin, y_pred_bin)
        f1_scores.append(f1)
    
    # Plot F1 scores
    plt.figure(figsize=(12, 8))
    bars = plt.bar(classes, f1_scores, color='skyblue')
    
    # Add value labels on top of bars
    for bar, score in zip(bars, f1_scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.2f}', ha='center', fontsize=10)
    
    plt.xlabel('Classes', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.title('F1 Scores by Class', fontsize=14)
    plt.ylim(0, 1.1)  # Set y-axis limit with a little margin for text
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    plt.tight_layout()
    
    # Save the figure
    plot_path = os.path.join(plots_dir, f"f1_scores_{run_name}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved F1 scores plot to {plot_path}")
    
    # Also save the F1 scores as CSV
    f1_df = pd.DataFrame({'Class': classes, 'F1_Score': f1_scores})
    f1_csv_path = os.path.join(plots_dir, f"f1_scores_{run_name}.csv")
    f1_df.to_csv(f1_csv_path, index=False)
    print(f"Saved F1 scores to {f1_csv_path}")
    
    plt.show()

def plot_roc_curves(y_val, y_pred_prob, classes, plots_dir, run_name):
    """Plot ROC curves for each class and save the figure."""
    n_classes = len(classes)
    
    # Convert to one-hot encoding for ROC curve
    y_val_bin = np.zeros((len(y_val), n_classes))
    for i in range(len(y_val)):
        y_val_bin[i, y_val[i]] = 1
    
    # Plot ROC curves
    plt.figure(figsize=(12, 8))
    
    # Store AUC scores for CSV export
    auc_scores = {}
    
    # For each class
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_val_bin[:, i], y_pred_prob[:, i])
        roc_auc = auc(fpr, tpr)
        auc_scores[classes[i]] = roc_auc
        
        plt.plot(fpr, tpr, lw=2, 
                 label=f'{classes[i]} (AUC={roc_auc:.2f})')
    
    # Plot diagonal line (random classifier)
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curves', fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the figure
    plot_path = os.path.join(plots_dir, f"roc_curves_{run_name}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved ROC curves to {plot_path}")
    
    # Save AUC scores as CSV
    auc_df = pd.DataFrame({'Class': list(auc_scores.keys()), 'AUC': list(auc_scores.values())})
    auc_csv_path = os.path.join(plots_dir, f"auc_scores_{run_name}.csv")
    auc_df.to_csv(auc_csv_path, index=False)
    print(f"Saved AUC scores to {auc_csv_path}")
    
    plt.show()

def evaluate_model(model, X_val, y_val, classes, plots_dir, run_name):
    """Evaluate the model and print detailed metrics."""
    print("\nModel Evaluation:")
    
    # Get predictions
    y_pred_prob = model.predict(X_val)
    y_pred_classes = np.argmax(y_pred_prob, axis=1)
    
    # Print classification report
    print("\nClassification Report:")
    report = classification_report(y_val, y_pred_classes, target_names=classes, output_dict=True)
    print(classification_report(y_val, y_pred_classes, target_names=classes))
    
    # Save classification report to CSV
    if isinstance(report, tf.Tensor):
        report = report.numpy()
    # Then create the DataFrame and transpose it
    report_df = pd.DataFrame(report).transpose()
    report_csv_path = os.path.join(plots_dir, f"classification_report_{run_name}.csv")
    report_df.to_csv(report_csv_path)
    print(f"Saved classification report to {report_csv_path}")
    
    # Plot and save confusion matrix with histogram
    plot_confusion_matrix_with_histogram(y_val, y_pred_classes, classes, plots_dir, run_name)
    
    # Plot and save F1 scores for each class
    plot_f1_scores(y_val, y_pred_classes, classes, plots_dir, run_name)
    
    # Plot and save precision-recall curves for each class
    plot_precision_recall_curves(y_val, y_pred_prob, classes, plots_dir)
    
    # Plot and save ROC curves for each class
    plot_roc_curves(y_val, y_pred_prob, classes, plots_dir, run_name)
    
    return report

def save_training_summary(output_dir, classes, class_counts, history, report, run_name):
    """Save a summary of training parameters and results to CSV."""
    # Create summary dictionary
    summary = {
        'Date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'Total Classes': len(classes),
        'Final Training Accuracy': history.history['accuracy'][-1],
        'Final Training Loss': history.history['loss'][-1],
        'Final Validation Accuracy': history.history['val_accuracy'][-1],
        'Final Validation Loss': history.history['val_loss'][-1],
        'Macro Avg Precision': report['macro avg']['precision'],
        'Macro Avg Recall': report['macro avg']['recall'],
        'Macro Avg F1-score': report['macro avg']['f1-score'],
        'Weighted Avg Precision': report['weighted avg']['precision'],
        'Weighted Avg Recall': report['weighted avg']['recall'],
        'Weighted Avg F1-score': report['weighted avg']['f1-score'],
    }
    
    # Add class counts
    for cls, count in class_counts.items():
        if cls in classes:
            summary[f'{cls}_count'] = count
    
    # Add per-class metrics
    for cls in classes:
        if cls in report:
            summary[f'{cls}_precision'] = report[cls]['precision']
            summary[f'{cls}_recall'] = report[cls]['recall']
            summary[f'{cls}_f1'] = report[cls]['f1-score']
    
    # Save to CSV
    summary_df = pd.DataFrame([summary])
    summary_path = os.path.join(output_dir, f"training_summary_{run_name}.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved training summary to {summary_path}")
    
    return summary

# Don't forget to add these imports at the top of your script
# from sklearn.metrics import roc_curve, auc, f1_score

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

def preprocess_and_augment(image_path, target_size=(224, 224), augment=True):
    """Load and preprocess a single image with optional augmentation."""
    img = Image.open(image_path).convert('L').resize(target_size, Image.Resampling.LANCZOS) # check add padding tf.pad
    img_array = np.array(img).reshape(*target_size, 1) / 255.0
    
    if not augment:
        # Return the same image three times if no augmentation
        return [img_array]
    
    # 1. Original image
    original = img_array
    
    # 2. Horizontally flipped version
    img_flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
    flipped = np.array(img_flipped).reshape(*target_size, 1) / 255.0
    
    return [original, flipped]

def preprocess_image(image_path, label, target_size=(224, 224), augment=False):
    """Preprocess a single image and apply optional augmentation."""
    # Convert tensor path to string properly
    try:
        if isinstance(image_path, tf.Tensor):
            image_path = image_path.numpy()
            
        if isinstance(image_path, np.ndarray):
            if image_path.dtype.type is np.str_:
                # It's a string in numpy array format
                image_path = str(image_path)
            elif image_path.dtype == np.object_:
                # It might be a numpy array of Python objects
                image_path = str(image_path.item())
            elif image_path.dtype == np.bytes_:
                # It's bytes, decode to string
                image_path = image_path.tobytes().decode('utf-8')
            else:
                # It's a numpy array containing bytes
                image_path = bytes(image_path).decode('utf-8')
        elif isinstance(image_path, bytes):
            # It's raw bytes, decode to string
            image_path = image_path.decode('utf-8')
            
        # Load and process image
        img = Image.open(image_path).convert('L').resize(target_size, Image.Resampling.LANCZOS)
        img_array = np.array(img).reshape(*target_size, 1) / 255.0
        
        # Convert to tensor for TF operations
        if augment:
            img_tensor = tf.convert_to_tensor(img_array)
            img_tensor = tf.image.random_flip_left_right(img_tensor)
            img_tensor = tf.image.random_brightness(img_tensor, lower=0.9, upper=1.1)
            img_tensor = tf.image.random_contrast(img_tensor, lower=0.8, upper=1.2)
            return img_tensor, label
        
        return tf.convert_to_tensor(img_array), label
        
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        print(f"Type of image_path: {type(image_path)}")
        if isinstance(image_path, np.ndarray):
            print(f"NumPy array dtype: {image_path.dtype}")
            print(f"NumPy array shape: {image_path.shape}")
        raise

def preprocess_image_tf(image_path, label, target_size=(224, 224), augment=False):
    """Wrapper for TensorFlow's `tf.py_function` to preprocess images."""
    # Use partial to fix the target_size and augment parameters
    def _preprocess_wrapper(path, lbl):
        return preprocess_image(path, lbl, target_size=target_size, augment=augment)
    
    # Only pass the path and label to py_function
    return tf.py_function(
        func=_preprocess_wrapper,
        inp=[image_path, label],
        Tout=(tf.float32, tf.int32)
    )

def create_tf_dataset(image_paths, labels, batch_size, augment=False):
    """Create a TensorFlow dataset from image paths and labels."""
    # Create the dataset from paths and labels (not loading images yet)
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    
    # Add caching before preprocessing to avoid repeatedly reading data
    dataset = dataset.cache()
    
    # Apply preprocessing on-demand (images are loaded one at a time)
    dataset = dataset.map(
        lambda x, y: preprocess_image_tf(x, y, augment=augment), 
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Add shuffling for training data
    if augment:  # Assuming augment=True means we're processing training data
        dataset = dataset.shuffle(buffer_size=min(len(image_paths), 1000))
    
    # Batch the dataset
    dataset = dataset.batch(batch_size)
   
   # This caches the preprocessed images to avoid redundant preprocessing
    dataset = dataset.cache()
    
    # Prefetch for performance - allows the pipeline to fetch data while model is training
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    dataset = dataset.__iter__()
    return dataset

from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_data_generators(X_train, X_val, y_train, y_val, batch_size=32):
    """Create data generators for training and validation using Keras ImageDataGenerator.
    
    This simplifies image loading and augmentation without complex preprocessing pipelines.
    """
    # Create training data generator with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Create validation data generator with just rescaling
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create flow_from_directory generators 
    # Note: Here we assume X_train and X_val are lists of file paths
    
    # Option 1: If data is already in memory as arrays (not recommended for large datasets)
    # train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)
    # val_generator = val_datagen.flow(X_val, y_val, batch_size=batch_size)
    
    # Option 2: For file paths, we need to create DataFrameIterator or use flow_from_directory
    # This is more memory efficient
    import pandas as pd
    
    # Convert integer labels to strings for categorical mode
    y_train_str = [str(y) for y in y_train]
    y_val_str = [str(y) for y in y_val]

	# Create DataFrames with paths and labels
    train_df = pd.DataFrame({'filename': X_train, 'class': y_train_str})
    val_df = pd.DataFrame({'filename': X_val, 'class': y_val_str})
    
    # Create generators from DataFrames
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col='filename',
        y_col='class',
        target_size=(224, 224),
        color_mode='grayscale',
        class_mode='categorical' if len(set(y_train)) > 2 else 'binary',
        batch_size=batch_size
    )
    
    val_generator = val_datagen.flow_from_dataframe(
        dataframe=val_df,
        x_col='filename',
        y_col='class',
        target_size=(224, 224),
        color_mode='grayscale',
        class_mode='categorical' if len(set(y_val)) > 2 else 'binary',
        batch_size=batch_size
    )
    
    return train_generator, val_generator


def train_model(base_dir, classes, output_dir, run_name, epochs=30, batch_size=32):
    """Main function to train the model."""
    print(f"Starting training process with {len(classes)} classes")
    print(f"Results will be saved to: {output_dir}")
    
    # Create output directories
    output_dir, plots_dir, models_dir = create_output_directory(output_dir, run_name)
    
    # Load and preprocess data
    X_train, X_val, y_train, y_val, class_counts = load_dataset(base_dir, classes, val_split=0.1)
    print(f"\nTraining set size: {len(X_train)} images")
    print(f"Validation set size: {len(X_val)} images")
    
    # Split the dataset into training and validation sets
    #X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    #print(f"\nTraining set size: {len(X_train)}")
    #print(f"Validation set size: {len(X_val)}")
    
    # Configure TensorFlow memory growth to avoid allocating all GPU memory at once
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"Memory growth setting failed: {e}")
    
    # Create generators with built-in caching and prefetching
    # Pass the classes parameter to the function
    train_generator, val_generator = create_data_generators(
        X_train, X_val, y_train, y_val, batch_size=batch_size
    )
    
    # Create and train model
    num_classes = len(classes)
    print(f"\nCreating model with {num_classes} output classes")
    model = create_model(num_classes)
    
    # Define Early Stopping and Model Checkpoint callbacks
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, mode='max', restore_best_weights=True, verbose=1)
    checkpoint = ModelCheckpoint(os.path.join(models_dir, f'best_model_{run_name}.h5'),
                                 monitor='val_accuracy',
                                 save_best_only=True)
    
    print("\nStarting model training...")
    
    # Calculate steps per epoch for better memory management
    steps_per_epoch = len(X_train) // batch_size
    validation_steps = len(X_val) // batch_size

	# add class Weight
    class_weight = {0: 2, 1: 1, 2: 1,3: 1,4: 1,5: 1,6: 1,7: 1,8: 1,9: 1}
    
    history = model.fit(
        train_generator,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_generator,
        validation_steps=validation_steps,
        callbacks=[early_stopping, checkpoint],# TensorBoard],
        verbose=1,
		class_weight=class_weight
    )
    
    print("\nTraining complete! Generating evaluation plots...")
    # We need to modify the evaluate_model function to work with generators
    # For now, let's use a simplified version
    
    # Get predictions batch by batch to avoid memory issues
    y_pred_probs = []
    y_true = []
    
    # Reset the generator
    val_generator.reset()
    
    # Predict in batches
    for i in range(validation_steps):
        x_batch, y_batch = next(val_generator)
        batch_preds = model.predict(x_batch)
        y_pred_probs.append(batch_preds)
        y_true.append(y_batch)
    
    # Concatenate batches
    y_pred_probs = np.vstack(y_pred_probs)
    y_true = np.vstack(y_true)
    
    # Convert from one-hot back to label indices
    y_true_indices = np.argmax(y_true, axis=1)
    y_pred_indices = np.argmax(y_pred_probs, axis=1)
    
    # Generate classification report
    report = classification_report(y_true_indices, y_pred_indices, 
                                  target_names=classes, output_dict=True)
    print("\nClassification Report:")
    print(classification_report(y_true_indices, y_pred_indices, target_names=classes))
    
    # Plot and save training history
    plot_training_history(history, plots_dir, run_name)
    
    # Plot confusion matrix
    plot_confusion_matrix_with_histogram(y_true_indices, y_pred_indices, classes, plots_dir, run_name)
    
    # Save model
    model_path = os.path.join(models_dir, f"fox_classifier_model_{run_name}.h5")
    model.save(model_path)
    print(f"Saved model to {model_path}")
    
    # Save training summary
    summary = save_training_summary(output_dir, classes, class_counts, history, report, run_name)
    return model, history, summary

# TensorBoard
#tf.keras.callbacks.TensorBoard(
#    log_dir= './logs',
#    histogram_freq=0,  # How often to log histogram visualizations
#    embeddings_freq=0,  # How often to log embedding visualizations
#    update_freq="epoch",
#) #call this from cmd line tensorboard --logdir=/full_path_to_your_logs

# Example usage:
if __name__ == "__main__":
    # Check if GPU is available
    using_gpu = setup_gpu()
    
    # Set batch size based on GPU availability
    if using_gpu:
        batch_size = 16  # Larger batch size for GPU
    else:
        batch_size = 4  # Smaller batch size for CPU
    
    # Data directory
    base_dir = "D:/Data/data_v15/Cleaned/"  # Change this to your data directory
    
    # Output directory for saving results
    output_dir = "C:/Users/c0062193/OneDrive - Newcastle University/General - Fox-AI/AI results/Model performance/"

	# Set a name for this training run
    run_name = f"fox_v17_cleaned_{timestamp}"  # Change this for each run
    
    # Class definitions
    classes = ['fox', 'person', 'badger', 'deer', 'bird'] #, 'squirrel',  'lagomorph', 'dog',  'deer', 'muntjack', 'boar']#, 'cat']
    
    # Train model
    model, history, summary = train_model(base_dir, classes, output_dir, epochs=100, run_name=run_name, batch_size=batch_size)
    
    print("\nTraining and evaluation completed successfully!")
    print(f"All results saved to {output_dir}")