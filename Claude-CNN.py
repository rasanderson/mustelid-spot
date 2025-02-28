import subprocess
import sys
import os

def install_requirements():
    """Install required packages if they are not already installed."""
    try:
        # Print current working directory for debugging
        print(f"Current working directory: {os.getcwd()}")
        
        # Get absolute path to requirements.txt
        requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
        print(f"Looking for requirements.txt at: {requirements_path}")
        
        # Check if file exists
        if not os.path.exists(requirements_path):
            raise FileNotFoundError(f"requirements.txt not found at {requirements_path}")
            
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', requirements_path])
        print("Required packages installed successfully!")
    except Exception as e:
        print(f"Error installing requirements: {str(e)}")
        sys.exit(1)

#if __name__ == '__main__':
#    install_requirements()

import numpy as np
from PIL import Image
import os
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from collections import Counter

import tensorflow as tf
import os

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

# Add this at the start of your main code
if __name__ == "__main__":
    using_gpu = setup_gpu()
    
    # You can then modify your model training parameters based on GPU availability
    if using_gpu:
        batch_size = 32  # Larger batch size for GPU
    else:
        batch_size = 16  # Smaller batch size for CPU


def load_and_preprocess_image(image_path, target_size=(224, 224)):
    """Load and preprocess a single image."""
    img = Image.open(image_path).convert('L').resize(target_size, Image.Resampling.LANCZOS)
    return np.array(img).reshape(*target_size, 1) / 255.0

def load_dataset(base_dir, classes):
    """Load images and labels from directory structure."""
    images = []
    labels = []
    class_counts = {cls: 0 for cls in classes}
    
    print(f"Starting to load dataset from {base_dir}")
    
    # Load fox images (class 0)
    fox_dir = Path(base_dir) / "Fox"
    print(f"\nLoading fox images from {fox_dir}")
    fox_count = 0
    for img_path in fox_dir.glob("**/*.jp*g"):
        try:
            images.append(load_and_preprocess_image(img_path))
            labels.append(0)
            fox_count += 1
            if fox_count % 100 == 0:
                print(f"Loaded {fox_count} fox images")
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
    class_counts['fox'] = fox_count
    
    # Load non-fox images (classes 1+)
    notfox_dir = Path(base_dir) / "Not Fox/preprocessed"
    print(f"\nLoading non-fox images from {notfox_dir}")
    
    for folder in notfox_dir.iterdir():
        if not folder.is_dir():
            continue
        
        try:
            # Extract the class name from the folder name
            folder_name = folder.name.lower()  # Convert to lowercase
            
            # Find which class this folder corresponds to
            class_found = False
            for class_idx, class_name in enumerate(classes):
                if class_name in folder_name:
                    print(f"\nProcessing {class_name} images from {folder}")
                    class_count = 0
                    
                    for img_path in folder.glob("**/*.jp*g"):
                        try:
                            images.append(load_and_preprocess_image(img_path))
                            labels.append(class_idx)
                            class_count += 1
                            if class_count % 100 == 0:
                                print(f"Loaded {class_count} {class_name} images")
                        except Exception as e:
                            print(f"Error loading {img_path}: {e}")
                    
                    class_counts[class_name] = class_count
                    print(f"Finished loading {class_count} {class_name} images")
                    class_found = True
                    break
            
            if not class_found:
                print(f"Could not find matching class for folder {folder}")
                
        except Exception as e:
            print(f"Error processing folder {folder}: {e}")
    
    print("\nDataset loading complete!")
    print("\nClass distribution:")
    for cls, count in class_counts.items():
        print(f"{cls}: {count} images")
    
    return np.array(images), np.array(labels)

def create_model(num_classes, input_shape=(224, 224, 1)):
    """Create and compile the CNN model."""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        Flatten(),
        Dense(224, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\nModel architecture:")
    model.summary()
    return model

def plot_training_history(history):
    """Plot training and validation metrics."""
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
    plt.show()

def plot_confusion_matrix_with_histogram(y_true, y_pred, classes):
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
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, ax=ax1, cmap="Blues")
    ax1.set_title('Raw Confusion Matrix', fontsize=14)
    ax1.set_ylabel('True Label', fontsize=12)
    ax1.set_xlabel('Predicted Label', fontsize=12)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    
    # Plot normalized values (as percentages)
    ax2 = fig.add_subplot(gs[1])
    sns.heatmap(cm_norm, annot=True, fmt='.1%', xticklabels=classes, yticklabels=classes, ax=ax2, cmap="Blues")
    ax2.set_title('Normalized Confusion Matrix', fontsize=14)
    ax2.set_ylabel('True Label', fontsize=12)
    ax2.set_xlabel('Predicted Label', fontsize=12)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
    
    # Plot class distribution histogram
    ax3 = fig.add_subplot(gs[2])
    bars = ax3.barh(class_names, counts, color='skyblue')
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
    plt.show()

def evaluate_model(model, X_val, y_val, classes):
    """Evaluate the model and print detailed metrics."""
    print("\nModel Evaluation:")
    
    # Get predictions
    y_pred_prob = model.predict(X_val)
    y_pred_classes = np.argmax(y_pred_prob, axis=1)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred_classes, target_names=classes))
    
    # Plot confusion matrix with histogram
    plot_confusion_matrix_with_histogram(y_val, y_pred_classes, classes)
    
    # Plot precision-recall curves for each class (if multi-class)
    n_classes = len(classes)
    if n_classes > 2:
        plt.figure(figsize=(12, 8))
        
        # Convert to one-hot encoding for precision-recall curve
        y_val_bin = np.zeros((len(y_val), n_classes))
        for i in range(len(y_val)):
            y_val_bin[i, y_val[i]] = 1
        
        # For each class
        for i in range(n_classes):
            precision, recall, _ = precision_recall_curve(y_val_bin[:, i], y_pred_prob[:, i])
            avg_precision = average_precision_score(y_val_bin[:, i], y_pred_prob[:, i])
            
            plt.plot(recall, precision, lw=2, 
                     label=f'{classes[i]} (AP={avg_precision:.2f})')
        
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curves for Each Class', fontsize=14)
        plt.legend(loc="best")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

def train_model(base_dir, classes, epochs=10, batch_size=32):
    """Main function to train the model."""
    print(f"Starting training process with {len(classes)} classes")
    
    # Load and preprocess data
    X, y = load_dataset(base_dir, classes)
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
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"\nTraining set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    
    # Create and train model using only the number of classes actually present
    num_classes = len(unique_classes)
    print(f"\nCreating model with {num_classes} output classes")
    model = create_model(num_classes)
    
    print("\nStarting model training...")
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        verbose=1
    )
    
    print("\nTraining complete! Generating evaluation plots...")
    
    # Plot training history
    plot_training_history(history)
    
    # Use the enhanced evaluate_model function directly
    evaluate_model(model, X_val, y_val, present_classes)
    
    return model, history
    
# Example usage:
if __name__ == "__main__":
    #base_dir = "C:/Users/nicho/OneDrive - Newcastle University/General - Fox-AI/Processed"
    base_dir = "C:/Users/c0062193.CAMPUS/OneDrive - Newcastle University/General - Fox-AI/Processed"
    classes = ['fox', 'person', 'bird', 'lagomorph', 'deer', 'squirrel', 'badger', 'dog']
    
    model, history = train_model(base_dir, classes)