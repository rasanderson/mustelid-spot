#Multi-class CNN training

# Import packages
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import glob
import os

# Define folders
fox_image_folder = 'C:/Users/nicho/OneDrive - Newcastle University/General - Fox-AI/Processed/Fox/'
notfox_image_folder = 'C:/Users/nicho/OneDrive - Newcastle University/General - Fox-AI/Processed/Not Fox/preprocessed/'

# Define valid classes
classes = ['person', 'bird', 'dog', 'lagomorph', 'deer', 'squirrel', 'badger', 'empty', 'cat']
image_paths = []
labels = []
class_paths = []
folders = []

def match_folder_by_class(folder_path, classes):
    # Get all folders in the directory
    folders = os.listdir(folder_path)
    
    # Dictionary to store folder-to-class mappings
    folder_class_map = {}
    
    for folder in folders:
        try:
            # Extract ID, assuming consistent naming convention
            folder_id = folder.split('_')[1]
            
            # Find matching class
            matching_class = classes[classes['id'] == folder_id]
            
            if not matching_class.empty:
                folder_class_map[folder] = matching_class['class_name'].iloc[1]
            else:
                print(f"No class found for folder: {folder}")
        
        except (IndexError, KeyError):
            print(f"Could not process folder: {folder}")
    
    return folder_class_map



def encode_classes(classes):
    """
    Bidirectional mapping of classes to numbers and vice versa
    Will take a list of classes and turn them into numerical values
    for training the CNN using enumerate.

    Args:
        list: list of classes

    Returns
        list: list with class as numbers, or vice versa
    """
    class_to_num = {cls: idx for idx, cls in enumerate(classes)}
    num_to_class = {idx: cls for idx, cls in enumerate(classes)}
    return class_to_num, num_to_class

class_to_num, num_to_class = encode_classes(classes)

# Example usage
print(class_to_num)

# Define function to create df of fox images
def get_image_paths(directory):
    """
    Get list of image paths from a directory.
    Supports both .jpg and .jpeg files.

    Args:
        directory (str): Path to image directory

    Returns:
        list: List of image file paths
    """
    dir_path = Path(directory)
    image_paths = []
    for pattern in ['**/*.jpg', '**/*.jpeg', '**/*.JPG', '**/*.JPEG']:
        image_paths.extend(dir_path.glob(pattern))


    image_paths = [str(path) for path in image_paths]
    print(f"Found {len(image_paths)} images in {directory}")
    return image_paths

def preprocess_image(image_path, target_size=(64, 64)):
    """
    Load, resize, and flatten an image to grayscale for CNN training.

    Args:
        image_path (str): Path to the image file
        target_size (tuple): Desired dimensions (height, width)

    Returns:
        numpy.ndarray: Preprocessed image as a numpy array
    """
    img = Image.open(image_path)
    img = img.convert('L')
    img = img.resize(target_size, Image.Resampling.LANCZOS)
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = img_array.reshape(target_size[0], target_size[1], 1)
    return img_array

def load_multi_class_data(fox_dir, non_fox_dir, target_size=(64,64)):
    """
    Load the preprocessed images into arrays with labels per class.
    
    Args:
        fox_dir (str): Path to directory containing fox images
        non_fox_dir (str): Path to directory containing non fox images seperated into seperate folders
    
    Returns:
        tuple: (X_data, y_labels) where X_data is the loaded images and y_labels are the respective class label
    """
    #Get paths
    fox_paths = get_image_paths(fox_image_folder)
    non_fox_paths = match_folder_by_class(non_fox_dir, classes)

    # Initiate array for image and label
    images = []
    labels = []

    # Process fox images (label 0)
    print("Loading fox images...")
    for i, path in enumerate(fox_paths):
        try:
            img = preprocess_image(path, target_size)
            images.append(img)
            labels.append(0)
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1} fox images")
        except Exception as e:
            print(f"Error processing {path}: {str(e)}")
        
    # Loop process non fox images (label 1-i)
    print("\nLoading non fox images...")
    for i, path in enumerate(match_folder_by_class(non_fox_paths)):
        try:
            img = preprocess_image(path, target_size)
            images.append(img)
            labels.append(i)
            print(f"{i} is class {path.name}")
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1} fox images")
        except Exception as e:
            print(f"Error processing {path}: {str(e)}")
        
     
    X_data = np.array(images)
    y_labels = np.array(labels)

    # Print dataset summary
    print("\nDataset summary:")
    print(f"Total images: {len(X_data)}")
    print(f"Fox images: {np.sum(y_labels == 1)}")
    print(f"Non-fox images: {np.sum(y_labels == 0)}")
    print(f"X shape: {X_data.shape}")
    print(f"y shape: {y_labels.shape}")

    return X_data, y_labels

# Call into a CNN
x_data, y_labels = load_multi_class_data(fox_image_folder, notfox_image_folder, target_size=(64,64))

# Split into training and validation
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(x_data, y_labels, test_size=0.2, random_state=42)

# Create and train model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential([
     Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid') #Change to multiclass equivalent
])

model.compile(optimizer = 'adam',
              loss = 'binary_crossentropy', #Change
              metrics = ['accuracy'])

# Train the model
history = model.fit(X_train, y_train,
                    epochs = 20,
                    batch_size = 32,
                    validation_data = (X_val, y_val))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

#CHECK THIS
def preprocess_images(base_path, classes_list):
    # Create class to number mapping
    class_to_num = {cls: idx for idx, cls in enumerate(classes_list)}
    
    image_paths = []
    labels_text = []
    labels_numeric = []
    
    # Iterate through folders
    for folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder)
        
        # Skip if not a directory
        if not os.path.isdir(folder_path):
            continue
        
        # Determine label from folder name
        # Assumes folder name matches one of the classes exactly
        if folder in classes_list:
            label_text = folder
            label_numeric = class_to_num[folder]
            
            # Collect image paths in this folder
            for image_file in os.listdir(folder_path):
                image_paths.append(os.path.join(folder_path, image_file))
                labels_text.append(label_text)
                labels_numeric.append(label_numeric)
    
    # Create DataFrame
    df = pd.DataFrame({
        'image_path': image_paths,
        'label_text': labels_text,
        'label_numeric': labels_numeric
    })
    
    return df