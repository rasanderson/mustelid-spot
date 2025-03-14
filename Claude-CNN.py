# CNN training with Claude's help for fox classification
# Nicholas Allen, SNES, Newcastle University

import os
import sys
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

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

import numpy as np
from pathlib import Path
from PIL import Image
import random

import numpy as np
from pathlib import Path
from PIL import Image
import random

def load_and_preprocess_image(image_path, target_size=(224, 224), augment=True):
    """Load and preprocess a single image with optional augmentation."""
    img = Image.open(image_path).convert('L').resize(target_size, Image.Resampling.LANCZOS)
    img_array = np.array(img).reshape(*target_size, 1) / 255.0
    
    if augment:
        # Create horizontally flipped version
        img_flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
        img_flipped_array = np.array(img_flipped).reshape(*target_size, 1) / 255.0
        
        # Return as a list of separate images rather than stacked array
        return [img_array, img_flipped_array]
    
    return img_array

def load_dataset(base_dir, classes, augment=True):
    """Load images and labels from directory structure with class balancing."""
    # Dictionary to store images by class
    class_images = {cls_idx: [] for cls_idx in range(len(classes))}
    class_counts = {cls: 0 for cls in classes}
    
    print(f"Starting to load dataset from {base_dir}")
    
    # Load fox images (class 0)
    fox_dir = Path(base_dir) / "Fox"
    print(f"\nLoading fox images from {fox_dir}")
    fox_count = 0
    for img_path in fox_dir.glob("**/*.jp*g"):
        # Only include images with 'crop' in the filename
        if 'crop' in img_path.name.lower():
            try:
                # Store the path instead of loading the image now
                class_images[0].append(img_path)
                fox_count += 1
                if fox_count % 100 == 0:
                    print(f"Found {fox_count} fox images")
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
    class_counts['fox'] = fox_count
    
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
            class_found = False
            for class_idx, class_name in enumerate(classes):
                if class_idx > 0 and class_name in folder_name:
                    print(f"\nProcessing {class_name} images from {folder}")
                    class_count = 0
                    
                    for img_path in folder.glob("**/*.jp*g") or folder.glob("**/*JP*G"):
                        try:
                            # Store the path instead of loading the image now
                            class_images[class_idx].append(img_path)
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
    # We'll use a flat list to collect all images
    all_images = []
    all_labels = []
    
    for class_idx, image_paths in class_images.items():
        # Randomly sample min_count images from this class
        sampled_paths = random.sample(image_paths, min_count)
        
        class_name = classes[class_idx] if class_idx > 0 else 'fox'
        print(f"Randomly sampled {min_count} images for class {class_name}")
        
        # Process images in batches to show progress
        batch_size = 100
        for batch_idx in range(0, len(sampled_paths), batch_size):
            batch_paths = sampled_paths[batch_idx:batch_idx + batch_size]
            batch_count = 0
            
            for img_path in batch_paths:
                try:
                    result = load_and_preprocess_image(img_path, augment=augment)
                    
                    # Handle both augmented and non-augmented cases
                    if augment and isinstance(result, list):
                        # For augmented data, add each image separately
                        for img in result:
                            all_images.append(img)
                            all_labels.append(class_idx)
                            batch_count += 1
                    else:
                        # For non-augmented data (or if augment=False)
                        all_images.append(result)
                        all_labels.append(class_idx)
                        batch_count += 1
                        
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
            
            print(f"  Processed batch {batch_idx//batch_size + 1}/{(len(sampled_paths)-1)//batch_size + 1} " +
                  f"({batch_count} images loaded)")
    
    # Convert to numpy arrays with proper shapes
    images_array = np.array(all_images)
    labels_array = np.array(all_labels)
    
    # Print information about the final dataset
    print("\nBalanced dataset loading complete!")
    imgs_per_class = min_count * (2 if augment else 1)
    print(f"Each class has exactly {imgs_per_class} images " + 
          f"({min_count} original + {min_count} augmented)" if augment else "")
    print(f"Total dataset size: {len(images_array)} images")
    print(f"Dataset shape: {images_array.shape}")
    
    # Create final class counts dictionary for return
    # Include augmentation in the counts if used
    final_class_counts = {
        classes[i] if i > 0 else 'fox': min_count * (2 if augment else 1) 
        for i in range(len(classes))
    }
    
    return images_array, labels_array, final_class_counts

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
        
        # Data augmentation layers (part of the model)
        #preprocessing.RandomFlip("horizontal"), #done in preprocessing now.
        preprocessing.RandomRotation(0.2),
        preprocessing.RandomZoom(0.2),
        preprocessing.RandomTranslation(0.1, 0.1),
		
		# Image standardization
		Lambda(lambda x: tf.image.per_image_standardization(x), input_shape=input_shape, name='standardization'),

        # First block - 32 filters
        Conv2D(32, (3, 3), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        Conv2D(32, (3, 3), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        MaxPooling2D((2, 2)),
        BatchNormalization(),
        Dropout(0.25),
        
        # Second block - 64 filters
        Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        MaxPooling2D((2, 2)),
        BatchNormalization(),
        Dropout(0.25),
        
        # Third block - 128 filters
        Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        MaxPooling2D((2, 2)),
        BatchNormalization(),
        Dropout(0.25),
        
        # Fully connected layers
        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy', #binary_crossentropy #sparse_categorical_crossentropy
        metrics=['accuracy']
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

def plot_precision_recall_curves(y_val, y_pred_prob, classes, plots_dir):
    """Plot precision-recall curves for each class and save the figure."""
    n_classes = len(classes)
    
    # Convert to one-hot encoding for precision-recall curve
    y_val_bin = np.zeros((len(y_val), n_classes))
    for i in range(len(y_val)):
        y_val_bin[i, y_val[i]] = 1
    
    plt.figure(figsize=(12, 8))
    
    # Store average precision scores for CSV export
    ap_scores = {}
    
    # For each class
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_val_bin[:, i], y_pred_prob[:, i])
        avg_precision = average_precision_score(y_val_bin[:, i], y_pred_prob[:, i])
        ap_scores[classes[i]] = avg_precision
        
        plt.plot(recall, precision, lw=2, 
                 label=f'{classes[i]} (AP={avg_precision:.2f})')
    
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curves for Each Class', fontsize=14)
    plt.legend(loc="best")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the figure
    plot_path = os.path.join(plots_dir, f"precision_recall_curves_{run_name}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved precision-recall curves to {plot_path}")
    
    # Save average precision scores as CSV
    ap_df = pd.DataFrame({'Class': list(ap_scores.keys()), 'Average_Precision': list(ap_scores.values())})
    ap_csv_path = os.path.join(plots_dir, f"average_precision_scores_{run_name}.csv")
    ap_df.to_csv(ap_csv_path, index=False)
    print(f"Saved average precision scores to {ap_csv_path}")
    
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
    report_df = pd.DataFrame(report).transpose()
    report_csv_path = os.path.join(plots_dir, f"classification_report_{run_name}.csv")
    report_df.to_csv(report_csv_path)
    print(f"Saved classification report to {report_csv_path}")
    
    # Plot and save confusion matrix with histogram
    plot_confusion_matrix_with_histogram(y_val, y_pred_classes, classes, plots_dir, run_name)
    
    # Plot and save precision-recall curves for each class (if multi-class)
    n_classes = len(classes)
    if n_classes > 2:
        plot_precision_recall_curves(y_val, y_pred_prob, classes, plots_dir)
    
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

def train_model(base_dir, classes, output_dir, run_name, epochs=100, batch_size=32):
    """Main function to train the model."""
    print(f"Starting training process with {len(classes)} classes")
    print(f"Results will be saved to: {output_dir}")
    
    # Create output directories
    output_dir, plots_dir, models_dir = create_output_directory(output_dir, run_name)
    
    # Load and preprocess data
    X, y, class_counts = load_dataset(base_dir, classes)
    print(f"\nTotal dataset size: {len(X)} images")
    
    # Find which classes are actually present in the dataset
    unique_classes = np.unique(y)
    present_classes = [classes[i] for i in unique_classes]
    
    print("\nClasses present in dataset:")
    for i, class_idx in enumerate(unique_classes):
        count = np.sum(y == class_idx)
        print(f"{classes[class_idx]}: {count} images (class index {class_idx})")
    
    print("\nMissing classes:")
    missing_classes = [cls for i, cls in enumerate(classes) if i not in unique_classes]
    for cls in missing_classes:
        print(f"- {cls}")
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
    print(f"\nTraining set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    
    # Create and train model using only the number of classes actually present
    num_classes = len(unique_classes)
    print(f"\nCreating model with {num_classes} output classes")
    model = create_model(num_classes)

	# Define Early Stopping and Model Checkpoint callbacks
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, mode='max', restore_best_weights=True, verbose=1)
    checkpoint = ModelCheckpoint(os.path.join(models_dir, f'best_model_{run_name}.h5'), 
                                monitor='val_accuracy', 
                                save_best_only=True)
    
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
    report = evaluate_model(model, X_val, y_val, present_classes, plots_dir, run_name)
    
    # Save model
    model_path = os.path.join(models_dir, f"fox_classifier_model_{run_name}.h5")
    model.save(model_path)
    print(f"Saved model to {model_path}")
    
    # Save training summary
    summary = save_training_summary(output_dir, present_classes, class_counts, history, report, run_name)
    
    return model, history, summary

# Example usage:
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
    run_name = f"fox_v9_{timestamp}"  # Change this for each run
    
    # Class definitions
    classes = ['fox', 'lagomorph', 'person', 'squirrel', 'badger', 'dog', 'bird', 'deer', 'muntjack', 'boar']#, 'cat']
    
    # Train model
    model, history, summary = train_model(base_dir, classes, output_dir, epochs=100, run_name=run_name, batch_size=batch_size)
    
    print("\nTraining and evaluation completed successfully!")
    print(f"All results saved to {output_dir}")