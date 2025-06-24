import os
import hashlib
from PIL import Image
import shutil
from collections import defaultdict

def get_image_hash(image_path):
    """Generate a hash of the image content"""
    try:
        with Image.open(image_path) as img:
            # Convert to RGB to handle different formats consistently
            img = img.convert('RGB')
            # Resize to small size for faster comparison
            img = img.resize((8, 8))
            # Get hash of pixel data
            return hashlib.md5(img.tobytes()).hexdigest()
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def get_all_images_from_path(path, use_subfolders=True):
    """Get all image files from a path, optionally including subfolders"""
    images = []
    
    if use_subfolders:
        # Walk through all subdirectories
        for root, dirs, files in os.walk(path):
            for filename in files:
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                    images.append(os.path.join(root, filename))
    else:
        # Only look in the direct path
        if os.path.exists(path):
            for filename in os.listdir(path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                    images.append(os.path.join(path, filename))
    
    return images

def remove_duplicates_in_folder(folder_path, folder_name, use_subfolders=True):
    """Remove duplicate images within a folder, optionally checking subfolders"""
    print(f"\n=== Checking for duplicates in {folder_name} ===")
    
    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} does not exist, skipping...")
        return
    
    if use_subfolders:
        # Get all subfolders (classes)
        classes = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
        
        total_duplicates_removed = 0
        
        for class_name in classes:
            class_path = os.path.join(folder_path, class_name)
            print(f"  Checking class: {class_name}")
            
            # Dictionary to store hash -> list of files with that hash
            hash_to_files = defaultdict(list)
            
            # Get all image files and their hashes
            images = get_all_images_from_path(class_path, use_subfolders=True)
            
            for image_path in images:
                filename = os.path.basename(image_path)
                hash_val = get_image_hash(image_path)
                if hash_val:
                    hash_to_files[hash_val].append((filename, image_path))
            
            # Find and remove duplicates
            duplicates_removed = 0
            for hash_val, files in hash_to_files.items():
                if len(files) > 1:
                    # Keep the first file, remove the rest
                    files_to_remove = files[1:]  # Skip the first one
                    print(f"    Found {len(files)} copies of same image:")
                    print(f"      Keeping: {files[0][0]}")
                    
                    for filename, file_path in files_to_remove:
                        print(f"      Removing: {filename}")
                        os.remove(file_path)
                        duplicates_removed += 1
            
            if duplicates_removed > 0:
                print(f"    Removed {duplicates_removed} duplicates from {class_name}")
            else:
                print(f"    No duplicates found in {class_name}")
            
            total_duplicates_removed += duplicates_removed
    
    else:
        # Treat as flat folder
        print(f"  Checking folder as flat structure")
        
        hash_to_files = defaultdict(list)
        images = get_all_images_from_path(folder_path, use_subfolders=False)
        
        for image_path in images:
            filename = os.path.basename(image_path)
            hash_val = get_image_hash(image_path)
            if hash_val:
                hash_to_files[hash_val].append((filename, image_path))
        
        # Find and remove duplicates
        total_duplicates_removed = 0
        for hash_val, files in hash_to_files.items():
            if len(files) > 1:
                files_to_remove = files[1:]
                print(f"  Found {len(files)} copies of same image:")
                print(f"    Keeping: {files[0][0]}")
                
                for filename, file_path in files_to_remove:
                    print(f"    Removing: {filename}")
                    os.remove(file_path)
                    total_duplicates_removed += 1
        
        if total_duplicates_removed == 0:
            print(f"  No duplicates found")
    
    print(f"Total duplicates removed from {folder_name}: {total_duplicates_removed}")

def find_unique_images_flexible(folder1, folder2, output_folder, 
                               folder1_use_subfolders=True, 
                               folder2_use_subfolders=True):
    """Find unique images with flexible subfolder handling"""
    
    print(f"\n=== Finding unique images between datasets ===")
    print(f"Folder1 subfolders: {'Yes' if folder1_use_subfolders else 'No'}")
    print(f"Folder2 subfolders: {'Yes' if folder2_use_subfolders else 'No'}")
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    if folder1_use_subfolders and folder2_use_subfolders:
        # Both use subfolders - original logic
        return find_unique_images_with_subfolders_original(folder1, folder2, output_folder)
    
    elif not folder1_use_subfolders and not folder2_use_subfolders:
        # Both are flat - simple comparison
        return find_unique_images_flat(folder1, folder2, output_folder)
    
    else:
        # Mixed structure - need special handling
        return find_unique_images_mixed(folder1, folder2, output_folder, 
                                       folder1_use_subfolders, folder2_use_subfolders)

def find_unique_images_flat(folder1, folder2, output_folder):
    """Handle case where both folders are flat (no subfolders)"""
    
    print("Processing as flat folders...")
    
    # Get all hashes from folder1
    folder1_hashes = set()
    folder1_images = get_all_images_from_path(folder1, use_subfolders=False)
    
    for image_path in folder1_images:
        hash_val = get_image_hash(image_path)
        if hash_val:
            folder1_hashes.add(hash_val)
    
    print(f"Found {len(folder1_images)} images in cleaned dataset")
    
    # Check folder2 images
    folder2_images = get_all_images_from_path(folder2, use_subfolders=False)
    moved_count = 0
    
    for image_path in folder2_images:
        hash_val = get_image_hash(image_path)
        filename = os.path.basename(image_path)
        
        if hash_val and hash_val not in folder1_hashes:
            # Unique image, copy it
            shutil.copy2(image_path, os.path.join(output_folder, filename))
            moved_count += 1
    
    print(f"Moved {moved_count} unique images from {len(folder2_images)} total")

def find_unique_images_mixed(folder1, folder2, output_folder, 
                            folder1_use_subfolders, folder2_use_subfolders):
    """Handle mixed structures (one has subfolders, one doesn't)"""
    
    print("Processing mixed folder structures...")
    
    # Get all hashes from folder1 (regardless of structure)
    folder1_hashes = set()
    folder1_images = get_all_images_from_path(folder1, use_subfolders=folder1_use_subfolders)
    
    for image_path in folder1_images:
        hash_val = get_image_hash(image_path)
        if hash_val:
            folder1_hashes.add(hash_val)
    
    print(f"Found {len(folder1_images)} images in cleaned dataset")
    
    # Get all images from folder2 (regardless of structure)
    folder2_images = get_all_images_from_path(folder2, use_subfolders=folder2_use_subfolders)
    moved_count = 0
    
    for image_path in folder2_images:
        hash_val = get_image_hash(image_path)
        filename = os.path.basename(image_path)
        
        if hash_val and hash_val not in folder1_hashes:
            # Unique image, copy it
            # Handle potential filename conflicts
            output_path = os.path.join(output_folder, filename)
            counter = 1
            while os.path.exists(output_path):
                name, ext = os.path.splitext(filename)
                output_path = os.path.join(output_folder, f"{name}_{counter}{ext}")
                counter += 1
            
            shutil.copy2(image_path, output_path)
            moved_count += 1
    
    print(f"Moved {moved_count} unique images from {len(folder2_images)} total")

def find_unique_images_with_subfolders_original(folder1, folder2, output_folder):
    """Original subfolder logic (both folders have matching class structure)"""
    
    # Get all subfolders (classes) in both folders
    folder1_classes = set([d for d in os.listdir(folder1) if os.path.isdir(os.path.join(folder1, d))])
    folder2_classes = set([d for d in os.listdir(folder2) if os.path.isdir(os.path.join(folder2, d))])
    
    # Find common classes
    common_classes = folder1_classes & folder2_classes
    folder2_only_classes = folder2_classes - folder1_classes
    folder1_only_classes = folder1_classes - folder2_classes
    
    print(f"Common classes: {len(common_classes)}")
    
    # Warn about mismatched classes
    if folder2_only_classes:
        print(f"\n⚠️  WARNING: Classes in original dataset but not in cleaned dataset:")
        for class_name in sorted(folder2_only_classes):
            print(f"   - {class_name} - SKIPPING")
    
    if folder1_only_classes:
        print(f"\n⚠️  WARNING: Classes in cleaned dataset but not in original dataset:")
        for class_name in sorted(folder1_only_classes):
            print(f"   - {class_name}")
    
    total_moved = 0
    
    # Process only common classes
    for class_name in sorted(common_classes):
        print(f"\nProcessing class: {class_name}")
        
        folder1_class_path = os.path.join(folder1, class_name)
        folder2_class_path = os.path.join(folder2, class_name)
        output_class_path = os.path.join(output_folder, class_name)
        
        os.makedirs(output_class_path, exist_ok=True)
        
        # Get hashes from folder1 (including all subfolders)
        folder1_hashes = set()
        folder1_images = get_all_images_from_path(folder1_class_path, use_subfolders=True)
        
        for image_path in folder1_images:
            hash_val = get_image_hash(image_path)
            if hash_val:
                folder1_hashes.add(hash_val)
        
        print(f"  Found {len(folder1_images)} images in cleaned {class_name}")
        
        # Check folder2 images (including all subfolders)
        folder2_images = get_all_images_from_path(folder2_class_path, use_subfolders=True)
        moved_count = 0
        
        for image_path in folder2_images:
            hash_val = get_image_hash(image_path)
            filename = os.path.basename(image_path)
            
            if hash_val and hash_val not in folder1_hashes:
                # Handle potential filename conflicts
                output_path = os.path.join(output_class_path, filename)
                counter = 1
                while os.path.exists(output_path):
                    name, ext = os.path.splitext(filename)
                    output_path = os.path.join(output_class_path, f"{name}_{counter}{ext}")
                    counter += 1
                
                shutil.copy2(image_path, output_path)
                moved_count += 1
        
        print(f"  Moved {moved_count} unique images from {class_name}")
        total_moved += moved_count
        
        if moved_count == 0:
            os.rmdir(output_class_path)
    
    print(f"\nTotal unique images moved: {total_moved}")

def process_datasets(folder1, folder2, output_folder, 
                    folder1_use_subfolders=True, 
                    folder2_use_subfolders=True,
                    remove_duplicates=True):
    """Complete workflow with flexible subfolder handling"""
    
    if remove_duplicates:
        print("STEP 1: Removing duplicates within each dataset")
        print("=" * 50)
        
        remove_duplicates_in_folder(folder1, "cleaned dataset", folder1_use_subfolders)
        remove_duplicates_in_folder(folder2, "original dataset", folder2_use_subfolders)
    
    print("\nSTEP 2: Finding unique images between datasets")
    print("=" * 50)
    
    find_unique_images_flexible(folder1, folder2, output_folder, 
                               folder1_use_subfolders, folder2_use_subfolders)
    
    print("\n" + "=" * 50)
    print("PROCESS COMPLETE!")
    print("=" * 50)

# Usage examples:
folder1 = "G:/Data/data_v15/Cleaned/Fox"  # Your cleaned dataset with subfolders
folder2 = "D:/Fox-AI/Original_data/fox"  # Your original dataset with subfolders
output_folder = "G:/Data/data_v15/Removed/Fox"  # Where to put unique images

process_datasets(folder1, folder2, output_folder, 
                folder1_use_subfolders=True,   # Cleaned has subclasses
                folder2_use_subfolders=False,  # Original is flat
                remove_duplicates=True)

# Example 2: Both are flat folders
# process_datasets(folder1, folder2, output_folder, 
#                 folder1_use_subfolders=False, 
#                 folder2_use_subfolders=False)

# Example 3: Skip duplicate removal
# process_datasets(folder1, folder2, output_folder, 
#                 folder1_use_subfolders=True, 
#                 folder2_use_subfolders=False,
#                 remove_duplicates=False)