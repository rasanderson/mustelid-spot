import random
import numpy as np
from pathlib import Path

def extract_test_set(base_dir, classes, test_split=0.1, val_split=0, seed=42):
    """
    Extract the same test set that would be created by your load_dataset function.
    This ensures reproducible test splits for validation work.
    """
    # Set seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    
    # Dictionary to store image paths by class
    class_images = {cls_idx: [] for cls_idx in range(len(classes))}
    
    print(f"Loading dataset from {base_dir}")
    
    # Load fox images (class 0)
    fox_dir = Path(base_dir) / "Fox"
    print(f"Loading fox images from {fox_dir}")
    fox_patterns = list(fox_dir.glob("**/*.jpg")) + list(fox_dir.glob("**/*.jpeg")) + \
                   list(fox_dir.glob("**/*.JPG")) + list(fox_dir.glob("**/*.JPEG")) + \
                   list(fox_dir.glob("**/*.png")) + list(fox_dir.glob("**/*.PNG"))
    
    for img_path in fox_patterns:
        if 'crop' in img_path.name.lower():
            class_images[0].append(str(img_path))
    
    # Load non-fox images (classes 1+)
    notfox_dir = Path(base_dir) / "Not Fox/preprocessed/"
    
    for folder in notfox_dir.iterdir():
        if not folder.is_dir():
            continue
        
        folder_name = folder.name.lower()
        for class_idx, class_name in enumerate(classes):
            if class_idx > 0 and class_name in folder_name:
                img_patterns = list(folder.glob("**/*.jpg")) + list(folder.glob("**/*.jpeg")) + \
                              list(folder.glob("**/*.JPG")) + list(folder.glob("**/*.JPEG")) + \
                              list(folder.glob("**/*.png")) + list(folder.glob("**/*.PNG"))
                for img_path in img_patterns:
                    class_images[class_idx].append(str(img_path))
                break
    
    # Balance classes (same as your original function)
    min_count = min([len(imgs) for imgs in class_images.values()])
    print(f"Balancing to {min_count} images per class")
    
    all_image_paths = []
    all_labels = []
    
    for class_idx, image_paths in class_images.items():
        # Randomly sample min_count images from this class
        sampled_paths = random.sample(image_paths, min_count)
        all_image_paths.extend(sampled_paths)
        all_labels.extend([class_idx] * len(sampled_paths))
    
    # Split dataset (exactly matching your original logic)
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
    
    # Extract test set
    X_test = [all_image_paths[i] for i in test_indices]
    y_test = [all_labels[i] for i in test_indices]
    
    print(f"Extracted {len(X_test)} test images")
    
    # Print class distribution in test set
    test_class_counts = {}
    for label in y_test:
        class_name = classes[label] if label > 0 else 'fox'
        test_class_counts[class_name] = test_class_counts.get(class_name, 0) + 1
    
    print("Test set class distribution:")
    for class_name, count in test_class_counts.items():
        print(f"  {class_name}: {count} images")
    
    return X_test, y_test

# Example usage:
if __name__ == "__main__":
    # Define your classes (same as in your main script)
    classes = ['fox', 'person', 'badger', 'deer', 'bird', 'squirrel', 'lagomorph']
    
    # Extract test set
    base_directory = "G:/Data/data_v15/Removed/"  # Update with your actual path
    test_images, test_labels = extract_test_set(
        base_dir=base_directory,
        classes=classes,
        test_split=0.1,
        val_split=0,
        seed=42
    )
    
    # Save test set paths to file for later use
    output_path = Path(base_directory) / 'test_set_paths_removed.txt'
    with open(output_path, 'w') as f:
        for img_path, label in zip(test_images, test_labels):
            class_name = classes[label] if label > 0 else 'fox'
            f.write(f"{img_path}\t{class_name}\n")
    
    print(f"Test set paths saved to 'test_set_paths.txt'")