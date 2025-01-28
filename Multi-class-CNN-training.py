#Multi-class CNN training

# Import packages


# Define folders
fox_image_folder = 'C:/Users/nicho/OneDrive - Newcastle University/General - Fox-AI/Processed/Fox/'
notfox_image_folder = 'C:/Users/nicho/OneDrive - Newcastle University/General - Fox-AI/Processed/Not Fox/Preprocessed/'

# Define valid classes
classes = ['person', 'bird', 'dog', 'lagomorph', 'deer', 'squirrel', 'badger', 'empty', 'cat']

# Define function to match class
def match_folder_by_class(folder_path, classes_df):
    # Extract folder name
    folder_name = Path(folder_path).name

    # Extract ID from folder name
    folder_id = folder_name.split('_')[1]

    # Look up id in classes
    matching_class = classes_df[classes_df['id'] == folder_id]

    if not matching_class.empty:
        return matching_class['class_name'].iloc[1]
    else:
        return None

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

def load_multi_class_data(fox_dir, non_fox_dir):
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
    non_fox_paths = match_folder_by_class(notfox_image_folder)

    # Initiate array for image and label
    images = []
    labels = []

    # Load fox images (label 0)
    print("Loading fox images...")
    for i, path in enumerate(fox_paths):
        try:
            img = Image.open(fox_paths)
            img_array = np.array(img)
            images.append(img)
            labels.append(0)
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1} fox images")
        except Exception as e:
            print(f"Error processing {path}: {str(e)}")
        




    # Define function to merge DataFrames

    return X_data, y_labels



# Call into a CNN