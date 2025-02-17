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

def load_and_preprocess_image(image_path, target_size=(64, 64)):
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

def create_model(num_classes, input_shape=(64, 64, 1)):
    """Create and compile the CNN model."""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
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

def plot_confusion_matrix(y_true, y_pred, classes):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def evaluate_model(model, X_val, y_val, classes):
    """Evaluate the model and print detailed metrics."""
    print("\nModel Evaluation:")
    
    # Get predictions
    y_pred = model.predict(X_val)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred_classes, target_names=classes))
    
    # Plot confusion matrix
    plot_confusion_matrix(y_val, y_pred_classes, classes)

def train_model(base_dir, classes, epochs=20, batch_size=32):
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
    
    # Evaluate model using only the classes that are present
    def evaluate_model(model, X_val, y_val, classes, present_class_indices):
        """Evaluate the model and print detailed metrics."""
        print("\nModel Evaluation:")
        
        # Get predictions
        y_pred = model.predict(X_val)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Use only the classes that are present in the dataset
        present_classes = [classes[i] for i in present_class_indices]
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_val, y_pred_classes, target_names=present_classes))
        
        # Plot confusion matrix
        plot_confusion_matrix(y_val, y_pred_classes, present_classes)
    
    evaluate_model(model, X_val, y_val, classes, unique_classes)
    
    return model, history

# Example usage:
if __name__ == "__main__":
    base_dir = "C:/Users/nicho/OneDrive - Newcastle University/General - Fox-AI/Processed"
    classes = ['fox', 'person', 'bird', 'dog', 'lagomorph', 'deer', 'squirrel', 'badger'] #, 'empty', 'cat']
    
    model, history = train_model(base_dir, classes)