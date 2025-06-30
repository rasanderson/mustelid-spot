import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.callbacks import TensorBoard
import datetime
import glob
import random
import shutil
from sklearn.model_selection import train_test_split

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

using_gpu = setup_gpu()

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Define image dimensions
IMG_SIZE = 224  # EfficientNetB0 expects 224x224 input

# Build the model with single-channel input adaptation
def build_model(num_classes=10):
    # Input for your single-channel images
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 1))
    
    # Convert single channel to three channels (duplicate across channels)
    x = layers.Conv2D(3, (1, 1), padding='same')(inputs)
    
    # Load pre-trained EfficientNetB0 without top layers
    base_model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base_model.trainable = False  # Freeze the base model initially
    
    # Connect our channel adapter to the base model
    x = base_model(x)
    
    # Add classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs, outputs)
    return model, base_model

# Define paths
base_data_dir = "F:/Fox-AI/Dataset/Cleaned/Not Fox/preprocessed"
balanced_data_dir = 'F:/Fox-AI/Balanced/'
os.makedirs(balanced_data_dir, exist_ok=True)

# Set the number of samples per class
samples_per_class = 3000
test_split = 0.2  # 20% for validation

# Class mapping
class_map = {
    'fox': 0, 'lagomorph': 1, 'person': 2, 'squirrel': 3, 'badger': 4,
    'dog': 5, 'bird': 6, 'deer': 7, 'muntjack': 8, 'boar': 9
}

# Create balanced dataset
for class_name in class_map.keys():
    # Create directories for this class
    train_dir = os.path.join(balanced_data_dir, 'train', class_name)
    val_dir = os.path.join(balanced_data_dir, 'val', class_name)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Find all images for this class using glob (searches recursively in subdirectories)
    class_images = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
	    # Search in directories that contain the class name
	    search_path = os.path.join(base_data_dir, f'**preprocessed_{class_name}_images**', '**', ext)
	    class_images.extend(glob.glob(search_path, recursive=True))
    
	    # Also search in standard class subdirectories in case some follow that pattern
	    std_search_path = os.path.join(base_data_dir, '**', class_name, '**', ext)
	    class_images.extend(glob.glob(std_search_path, recursive=True))
    
    print(f"Found {len(class_images)} images for class {class_name}")
    
    # If we have more than samples_per_class, randomly sample
    if len(class_images) > samples_per_class:
        class_images = random.sample(class_images, samples_per_class)
    
    # Split into train and validation
    train_images, val_images = train_test_split(
        class_images, test_size=test_split, random_state=42
    )
    
    # Copy images to their respective directories

    for img_path in train_images:
        try:
            dest_path = os.path.join(val_dir, os.path.basename(img_path))
            shutil.copy(img_path, dest_path)
        except FileNotFoundError:
            print(f"Warning: Could not find file {img_path}. Skipping.")
            continue
        except Exception as e:
            print(f"Error copying {img_path}: {str(e)}. Skipping.")
            continue
    
    for img_path in val_images:
        try:
            dest_path = os.path.join(val_dir, os.path.basename(img_path))
            shutil.copy(img_path, dest_path)
        except FileNotFoundError:
            print(f"Warning: Could not find file {img_path}. Skipping.")
            continue
        except Exception as e:
            print(f"Error copying {img_path}: {str(e)}. Skipping.")
            continue
    
    print(f"Copied {len(train_images)} training images and {len(val_images)} validation images for {class_name}")


# Now set up the generators to use the balanced dataset
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

# Training generator using the balanced dataset
train_generator = train_datagen.flow_from_directory(
    os.path.join(balanced_data_dir, 'train'),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=32,
    class_mode='categorical',
    color_mode='grayscale',
    classes=list(class_map.keys())
)

# Validation generator using the balanced dataset
val_generator = val_datagen.flow_from_directory(
    os.path.join(balanced_data_dir, 'val'),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=32,
    class_mode='categorical',
    color_mode='grayscale',
    classes=list(class_map.keys())
)

# Create and compile the model
model, base_model = build_model(num_classes=len(class_map))
model.compile(
    optimizer=optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Stage 1: Train only the top layers (which were randomly initialized)
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=10,  # Initial training to adjust the top layers
    validation_data=val_generator,
    validation_steps=val_generator.samples // val_generator.batch_size,
	callbacks=[tensorboard_callback]
)

# Stage 2: Fine-tune some top layers of the base model
# Unfreeze the top layers of the base model
base_model.trainable = True
for layer in base_model.layers[:-20]:  # Keep last 20 layers trainable
    layer.trainable = False

# Compile with a lower learning rate
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-5),  # Much lower learning rate
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Continue training
history_fine = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=10,  # Additional fine-tuning epochs
    validation_data=val_generator,
    validation_steps=val_generator.samples // val_generator.batch_size,
	callbacks=[tensorboard_callback]
)

# Save the model
model.save('transfer_classification_model.h5')