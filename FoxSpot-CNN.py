# Nicholas Allen
# MEP, Newcastle University
# 28/05/2025

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
#from tensorflow.keras.metrics import SparseCategoricalAccuracy, F1Score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Resizing, Lambda
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_recall_curve, average_precision_score 
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

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

def load_dataset(base_dir, classes, augment=True, val_split=0.2, test_split=0.1):
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
    
    # Set seed for reproducible splits
    np.random.seed(42)  # You can use any integer as the seed
    random.seed(42)

    # Split into train, validation, and test sets
    indices = list(range(len(all_image_paths)))
    random.shuffle(indices)

    # First split off test set
    test_split_idx = int(len(indices) * test_split)
    test_indices = indices[:test_split_idx]
    remaining_indices = indices[test_split_idx:]

    # Then split remaining into train and validation
    val_split_idx = int(len(remaining_indices) * val_split)
    val_indices = remaining_indices[:val_split_idx]
    train_indices = remaining_indices[val_split_idx:]
    
    X_train = [all_image_paths[i] for i in train_indices]
    y_train = [all_labels[i] for i in train_indices]
    X_val = [all_image_paths[i] for i in val_indices]
    y_val = [all_labels[i] for i in val_indices]
    X_test = [all_image_paths[i] for i in test_indices]
    y_test = [all_labels[i] for i in test_indices]
    
    # Create final class counts dictionary 
    final_class_counts = {
        classes[i] if i > 0 else 'fox': min_count * (2 if augment else 1)
        for i in range(len(classes))
    }
    
    print(f"\nSplit dataset: {len(X_train)} training, {len(X_val)} validation, {len(X_test)} test images")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, final_class_counts

# EVALUATION PLOTS
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score
from sklearn.preprocessing import label_binarize
from itertools import cycle

def plot_multiclass_evaluation(y_true, y_pred_proba, class_names, figsize=(15, 5)):
    """
    Plot ROC, Precision-Recall, and F1-score curves for multi-class classification.
    
    Parameters:
    y_true: array-like, true class labels (integers)
    y_pred_proba: array-like, predicted probabilities for each class
    class_names: list, names of the classes
    figsize: tuple, figure size
    """
    
    n_classes = len(class_names)
    
    # Binarize the output for multi-class ROC
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    if n_classes == 2:
        y_true_bin = np.column_stack([1 - y_true_bin, y_true_bin])
    
    # Colors for each class
    colors = cycle(['blue', 'red', 'orange', 'purple', 'green', 'brown', 'pink', 'gray'])
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # 1. ROC Curve
    ax1 = axes[0]
    for i, color in zip(range(n_classes), colors):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)
        ax1.plot(fpr, tpr, color=color, lw=2,
                label=f'{class_names[i]} (AUC = {roc_auc:.2f})')
    
    ax1.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.5)
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve')
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)
    
    # 2. Precision-Recall Curve
    ax2 = axes[1]
    colors = cycle(['blue', 'red', 'orange', 'purple', 'green', 'brown', 'pink', 'gray'])
    for i, color in zip(range(n_classes), colors):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_pred_proba[:, i])
        pr_auc = auc(recall, precision)
        ax2.plot(recall, precision, color=color, lw=2,
                label=f'{class_names[i]} (AUC = {pr_auc:.2f})')
    
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)
    
    # 3. F1-Score vs Threshold
    ax3 = axes[2]
    colors = cycle(['blue', 'red', 'orange', 'purple', 'green', 'brown', 'pink', 'gray'])
    for i, color in zip(range(n_classes), colors):
        # Calculate F1 scores for different thresholds
        precision, recall, thresholds = precision_recall_curve(y_true_bin[:, i], y_pred_proba[:, i])
    
        # Calculate F1 score for each threshold
        f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-8)
    
        # Use actual threshold values instead of indices
        ax3.plot(thresholds, f1_scores, color=color, lw=2,
                label=f'{class_names[i]}')

    ax3.set_xlim([0.0, 1.0])  # Set x-axis limits to 0-1 for thresholds
    ax3.set_ylim([0.0, 1.05])
    ax3.set_xlabel('Threshold')
    ax3.set_ylabel('F1 Score')
    ax3.set_title('F1 Score vs Threshold')
    ax3.legend(loc="upper right")
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

    return fig

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
    
    plt.tight_layout() #23,1cm x 63.8cm
    
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
        loss='sparse_categorical_crossentropy', #binary_crossentropy #sparse_categorical_crossentropy
        metrics=['accuracy', tf.keras.metrics.SparseCategoricalAccuracy()] #, tf.keras.metrics.F1Score(threshold=0.5)]
    )
    
    print("\nModel architecture:")
    model.summary()
    return model

def create_data_generators(X_train, X_val, X_test, y_train, y_val, y_test, batch_size=32):
    """Create data generators for training, validation, and testing using Keras ImageDataGenerator.
    
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
    
    # Create validation and test data generators with just rescaling (no augmentation)
    val_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create flow_from_directory generators 
    # Note: Here we assume X_train, X_val, and X_test are lists of file paths
    
    # Option 1: If data is already in memory as arrays (not recommended for large datasets)
    # train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)
    # val_generator = val_datagen.flow(X_val, y_val, batch_size=batch_size)
    # test_generator = test_datagen.flow(X_test, y_test, batch_size=batch_size)
    
    # Option 2: For file paths, we need to create DataFrameIterator or use flow_from_directory
    # This is more memory efficient
    import pandas as pd
    
    # Convert integer labels to strings for categorical mode
    y_train_str = [str(y) for y in y_train]
    y_val_str = [str(y) for y in y_val]
    y_test_str = [str(y) for y in y_test]
    
    # Create DataFrames with paths and labels
    train_df = pd.DataFrame({'filename': X_train, 'class': y_train_str})
    val_df = pd.DataFrame({'filename': X_val, 'class': y_val_str})
    test_df = pd.DataFrame({'filename': X_test, 'class': y_test_str})
    
    # Determine class mode based on number of classes
    num_classes = len(set(y_train))
    class_mode = 'sparse' if num_classes > 2 else 'binary'
    
    # Create generators from DataFrames
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col='filename',
        y_col='class',
        target_size=(224, 224),
        color_mode='grayscale',
        class_mode=class_mode,
        batch_size=batch_size,
        shuffle=True  # Shuffle training data
    )
    
    val_generator = val_datagen.flow_from_dataframe(
        dataframe=val_df,
        x_col='filename',
        y_col='class',
        target_size=(224, 224),
        color_mode='grayscale',
        class_mode=class_mode,
        batch_size=batch_size,
        shuffle=False  # Don't shuffle validation data
    )
    
    test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        x_col='filename',
        y_col='class',
        target_size=(224, 224),
        color_mode='grayscale',
        class_mode=class_mode,
        batch_size=batch_size,
        shuffle=False  # Don't shuffle test data
    )
    
    return train_generator, val_generator, test_generator

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

def train_model(base_dir, classes, output_dir, run_name, epochs=500, batch_size=32, val_split=0.2, test_split=0.1):
    """Main function to train the model."""
    print(f"Starting training process with {len(classes)} classes")
    print(f"Results will be saved to: {output_dir}")
    
    # Create output directories
    output_dir, plots_dir, models_dir = create_output_directory(output_dir, run_name)
    
    # Load and preprocess data
    X_train, X_val, X_test, y_train, y_val, y_test, class_counts = load_dataset(
        base_dir, classes, val_split=val_split, test_split=test_split
    )
    print(f"\nTraining set size: {len(X_train)} images")
    print(f"Validation set size: {len(X_val)} images")
    print(f"Test set size: {len(X_test)} images")
    
    # Configure TensorFlow memory growth to avoid allocating all GPU memory at once
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"Memory growth setting failed: {e}")
    
    # Create generators with built-in caching and prefetching
    train_generator, val_generator, test_generator = create_data_generators(
        X_train, X_val, X_test, y_train, y_val, y_test, batch_size=batch_size
    )
    
    # ========================================
    # MODEL CREATION AND TRAINING
    # ========================================
    
    # Create and train model
    num_classes = len(classes)
    print(f"\nCreating model with {num_classes} output classes")
    model = create_model(num_classes)
    
    # Define Early Stopping and Model Checkpoint callbacks
    early_stopping = EarlyStopping(monitor='val_sparse_categorical_accuracy', patience=10, mode='max', restore_best_weights=True, verbose=1)
    
    # Best model checkpoint (saves only when validation metric improves)
    best_checkpoint = ModelCheckpoint(
        os.path.join(models_dir, f'best_model_{run_name}.h5'),
        monitor='val_sparse_categorical_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    # Latest model checkpoint (saves after EVERY epoch as backup)
    latest_checkpoint = ModelCheckpoint(
        os.path.join(models_dir, f'latest_model_{run_name}.h5'),
        save_best_only=False,  # Save every epoch
        save_freq='epoch',     # Save frequency
        verbose=0
    )
    
    print("\nStarting model training...")
    
    # Calculate steps per epoch for better memory management
    steps_per_epoch = len(X_train) // batch_size
    validation_steps = len(X_val) // batch_size
    test_steps = len(X_test) // batch_size
    
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Validation steps: {validation_steps}")
    print(f"Test steps: {test_steps}")

    # Add class weight
    class_weight = {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1.5, 7: 1, 8: 1, 9: 1}
    
    # TRAIN THE MODEL
    history = model.fit(
        train_generator,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_generator,
        validation_steps=validation_steps,
        callbacks=[early_stopping, best_checkpoint, latest_checkpoint],
        verbose=1#,
        #class_weight=class_weight
    )
    
    print("\nTraining complete! Auto-saving final model as backup...")
    
    # Additional safety: Save the final model immediately after training
    final_model_path = os.path.join(models_dir, f"final_model_{run_name}.h5")
    model.save(final_model_path)
    print(f"Final model saved to: {final_model_path}")
    
    # Plot and save training history
    plot_training_history(history, plots_dir, run_name)
    
    # ========================================
    # VALIDATION SET EVALUATION
    # ========================================
    
    print("\n" + "="*50)
    print("EVALUATING ON VALIDATION SET")
    print("="*50)
    
    try:
        # Method 1: Using model.evaluate for quick metrics on validation set
        print("Running model.evaluate on validation set...")
        val_generator.reset()  # Reset before evaluation
        val_results_eval = model.evaluate(val_generator, steps=validation_steps, verbose=1)
        
        # Handle multiple metrics being returned
        if isinstance(val_results_eval, list):
            val_loss = val_results_eval[0]
            val_accuracy = val_results_eval[1] if len(val_results_eval) > 1 else val_results_eval[0]
            print(f"Validation Loss: {val_loss:.4f}")
            print(f"Validation Accuracy: {val_accuracy:.4f}")
            
            # Print all metrics if there are more than 2
            if len(val_results_eval) > 2:
                metric_names = model.metrics_names if hasattr(model, 'metrics_names') else [f"metric_{i}" for i in range(len(val_results_eval))]
                for i, (name, value) in enumerate(zip(metric_names, val_results_eval)):
                    print(f"Validation {name}: {value:.4f}")
        else:
            val_loss = val_results_eval
            val_accuracy = val_results_eval
            print(f"Validation Loss/Accuracy: {val_loss:.4f}")
        
        # Method 2: Getting detailed predictions for plots and classification report
        print("Generating detailed validation predictions...")
        
        # Reset validation generator for detailed predictions
        val_generator.reset()
        
        # Get predictions and true labels for validation set
        y_pred_probs_val = []
        y_true_val = []
        
        print(f"Processing {validation_steps} validation batches...")
        for i in range(validation_steps):
            try:
                x_batch, y_batch = next(val_generator)
                batch_preds = model.predict(x_batch, verbose=0)
                y_pred_probs_val.append(batch_preds)
                y_true_val.append(y_batch)
                
                # Debug prints for first batch
                if i == 0:
                    print(f"First batch - x_batch shape: {x_batch.shape}")
                    print(f"First batch - y_batch shape: {y_batch.shape}")
                    print(f"First batch - batch_preds shape: {batch_preds.shape}")
                
                if (i + 1) % 10 == 0:
                    print(f"Processed {i + 1}/{validation_steps} validation batches")
                    
            except StopIteration:
                print(f"Generator exhausted at batch {i}")
                break
        
        if not y_pred_probs_val or not y_true_val:
            raise ValueError("No validation predictions were collected")
        
        # Concatenate batches
        y_pred_probs_val = np.vstack(y_pred_probs_val)
        y_true_val = np.concatenate(y_true_val, axis=0)
        
        print(f"Validation predictions shape: {y_pred_probs_val.shape}")
        print(f"Validation true labels shape: {y_true_val.shape}")
        
        # Check if y_true_val is already indices or one-hot encoded
        if len(y_true_val.shape) == 1:
            # Already indices
            y_true_indices_val = y_true_val.astype(int)
        else:
            # One-hot encoded, convert to indices
            y_true_indices_val = np.argmax(y_true_val, axis=1)
        
        y_pred_indices_val = np.argmax(y_pred_probs_val, axis=1)
        
        print(f"Validation true indices shape: {y_true_indices_val.shape}")
        print(f"Validation pred indices shape: {y_pred_indices_val.shape}")
        
        # Generate classification report for validation set
        report_val = classification_report(y_true_indices_val, y_pred_indices_val, 
                                          target_names=classes, output_dict=True)
        print("\nValidation Classification Report:")
        print(classification_report(y_true_indices_val, y_pred_indices_val, target_names=classes))
        
        # GENERATE THE ROC/PR/F1 PLOTS FOR VALIDATION SET
        print("\nGenerating validation evaluation plots (ROC, PR, F1)...")
        fig_val = plot_multiclass_evaluation(y_true_indices_val, y_pred_probs_val, classes, figsize=(15, 5))
        
        # Save the validation evaluation plots
        fig_val.savefig(os.path.join(plots_dir, f'val_evaluation_curves_{run_name}.png'),
                    dpi=300, bbox_inches='tight')
        
        # Plot confusion matrix for validation set
        plot_confusion_matrix_with_histogram(y_true_indices_val, y_pred_indices_val, classes, plots_dir, run_name + "_val")
        
        val_results = (y_true_indices_val, y_pred_probs_val, y_pred_indices_val)

    except Exception as e:
        print(f"\nERROR during validation evaluation: {e}")
        print("Model training completed successfully, but validation evaluation failed.")
        
        # Set default values for failed validation evaluation
        val_accuracy = None
        report_val = None
        val_results = None
        
        print("Proceeding to test evaluation...")
    
    # ========================================
    # TEST SET EVALUATION
    # ========================================
    
    print("\n" + "="*50)
    print("EVALUATING ON TEST SET")
    print("="*50)
    
    try:
        # Method 1: Using model.evaluate for quick metrics on test set
        print("Running model.evaluate on test set...")
        test_generator.reset()  # Reset before evaluation
        test_results_eval = model.evaluate(test_generator, steps=test_steps, verbose=1)
        
        # Handle multiple metrics being returned
        if isinstance(test_results_eval, list):
            test_loss = test_results_eval[0]
            test_accuracy = test_results_eval[1] if len(test_results_eval) > 1 else test_results_eval[0]
            print(f"Test Loss: {test_loss:.4f}")
            print(f"Test Accuracy: {test_accuracy:.4f}")
            
            # Print all metrics if there are more than 2
            if len(test_results_eval) > 2:
                metric_names = model.metrics_names if hasattr(model, 'metrics_names') else [f"metric_{i}" for i in range(len(test_results_eval))]
                for i, (name, value) in enumerate(zip(metric_names, test_results_eval)):
                    print(f"Test {name}: {value:.4f}")
        else:
            test_loss = test_results_eval
            test_accuracy = test_results_eval
            print(f"Test Loss/Accuracy: {test_loss:.4f}")
        
        # Method 2: Getting detailed predictions for plots and classification report
        print("Generating detailed test predictions...")
        
        # Reset test generator for detailed predictions
        test_generator.reset()
        
        # Get predictions and true labels for test set
        y_pred_probs_test = []
        y_true_test = []
        
        print(f"Processing {test_steps} test batches...")
        for i in range(test_steps):
            try:
                x_batch, y_batch = next(test_generator)
                batch_preds = model.predict(x_batch, verbose=0)
                y_pred_probs_test.append(batch_preds)
                y_true_test.append(y_batch)
                
                # Debug prints for first batch
                if i == 0:
                    print(f"First batch - x_batch shape: {x_batch.shape}")
                    print(f"First batch - y_batch shape: {y_batch.shape}")
                    print(f"First batch - batch_preds shape: {batch_preds.shape}")
                
                if (i + 1) % 10 == 0:
                    print(f"Processed {i + 1}/{test_steps} test batches")
                    
            except StopIteration:
                print(f"Generator exhausted at batch {i}")
                break
        
        if not y_pred_probs_test or not y_true_test:
            raise ValueError("No test predictions were collected")
        
        # Concatenate batches
        y_pred_probs_test = np.vstack(y_pred_probs_test)
        y_true_test = np.concatenate(y_true_test, axis=0)
        
        print(f"Test predictions shape: {y_pred_probs_test.shape}")
        print(f"Test true labels shape: {y_true_test.shape}")
        
        # Check if y_true_test is already indices or one-hot encoded
        if len(y_true_test.shape) == 1:
            # Already indices
            y_true_indices_test = y_true_test.astype(int)
        else:
            # One-hot encoded, convert to indices
            y_true_indices_test = np.argmax(y_true_test, axis=1)
        
        y_pred_indices_test = np.argmax(y_pred_probs_test, axis=1)
        
        print(f"Test true indices shape: {y_true_indices_test.shape}")
        print(f"Test pred indices shape: {y_pred_indices_test.shape}")
        
        # Generate classification report for test set
        report_test = classification_report(y_true_indices_test, y_pred_indices_test, 
                                           target_names=classes, output_dict=True)
        print("\nTest Classification Report:")
        print(classification_report(y_true_indices_test, y_pred_indices_test, target_names=classes))
        
        # GENERATE THE ROC/PR/F1 PLOTS FOR TEST SET
        print("\nGenerating test evaluation plots (ROC, PR, F1)...")
        fig_test = plot_multiclass_evaluation(y_true_indices_test, y_pred_probs_test, classes, figsize=(15, 5))
        
        # Save the test evaluation plots
        fig_test.savefig(os.path.join(plots_dir, f'test_evaluation_curves_{run_name}.png'),
                    dpi=300, bbox_inches='tight')
        
        # Plot confusion matrix for test set
        plot_confusion_matrix_with_histogram(y_true_indices_test, y_pred_indices_test, classes, plots_dir, run_name + "_test")
        
        test_results = (y_true_indices_test, y_pred_probs_test, y_pred_indices_test)

    except Exception as e:
        print(f"\nERROR during test evaluation: {e}")
        print("Model training completed successfully, but test evaluation failed.")
        print("Your trained model has been saved and is available for use.")
        
        # Set default values for failed test evaluation
        test_accuracy = None
        report_test = None
        test_results = None
        
        print("Proceeding with available results...")
    
    # ========================================
    # FINAL MODEL SAVING AND SUMMARY
    # ========================================
    
    # Save model (final backup)
    model_path = os.path.join(models_dir, f"fox_classifier_model_{run_name}.h5")
    model.save(model_path)
    print(f"Saved final model to {model_path}")
    
    # Save training summary with test results (only if test evaluation succeeded)
    if 'report_test' in locals() and report_test is not None and 'test_accuracy' in locals():
        summary = save_training_summary_with_test(output_dir, classes, class_counts, history, 
                                                 report_test, test_accuracy, run_name)
    else:
        print("Skipping training summary with test results due to test evaluation failure")
        summary = None
    
    # Return results - ensure all variables exist
    val_results = val_results if 'val_results' in locals() else None
    test_results = test_results if 'test_results' in locals() else None
    
    return model, history, summary, val_results, test_results

def save_training_summary_with_test(output_dir, classes, class_counts, history, 
                                   report_val, report_test, test_accuracy, run_name):
    """Save training summary including test results."""
    summary = {
        'run_name': run_name,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'classes': classes,
        'class_counts': class_counts,
        'final_val_accuracy': max(history.history.get('val_sparse_categorical_accuracy', [0])),
        'final_val_loss': min(history.history.get('val_loss', [float('inf')])),
        'test_accuracy': test_accuracy,
        'validation_report': report_val,
        'test_report': report_test,
        'epochs_trained': len(history.history['loss'])
    }
    
    # Save to JSON
    summary_path = os.path.join(output_dir, f'training_summary_{run_name}.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4, default=str)
    
    print(f"Training summary saved to {summary_path}")
    return summary

# Updated example usage:
if __name__ == "__main__":
    # Check if GPU is available
    using_gpu = setup_gpu()
    
    # Set batch size based on GPU availability
    if using_gpu:
        batch_size = 16  # Larger batch size for GPU
    else:
        batch_size = 4  # Smaller batch size for CPU
    
    # Data directory
    base_dir = "G:/Data/data_v16/"  # Change this to your data directory
    
    # Output directory for saving results
    output_dir = "C:/Users/c0062193.CAMPUS/OneDrive - Newcastle University/General - Fox-AI/AI results/Model performance/"

    # Set a name for this training run
    run_name = f"fox_v21_newdata_{timestamp}"  # Change this for each run
    
    # Class definitions
    classes = ['fox', 'person', 'badger', 'deer', 'bird', 'squirrel', 'lagomorph']
    
    # Train model with test evaluation
    model, history, summary, test_results = train_model(
        base_dir, classes, output_dir, epochs=500, run_name=run_name, 
        batch_size=batch_size, val_split=0.2, test_split=0.1
    )
    
    # Extract test results if needed later
    y_true_test, y_pred_probs_test, y_pred_test = test_results
