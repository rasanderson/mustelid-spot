import os
import shutil
import argparse
import fnmatch

def copy_crop_images(source_dir, dest_dir):
    # Create destination directory if it doesn't exist
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    
    # Track statistics
    copied_count = 0
    skipped_count = 0
    
    # Walk through the source directory
    for root, dirs, files in os.walk(source_dir):
        # Filter for image files containing "crop"
        image_files = [f for f in files if "crop" in f.lower() and f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'))]
        
        if image_files:
            # Create relative path
            rel_path = os.path.relpath(root, source_dir)
            dest_path = os.path.join(dest_dir, rel_path)
            
            # Create destination subdirectory if needed
            if not os.path.exists(dest_path):
                os.makedirs(dest_path)
            
            # Copy each image file only if it doesn't already exist
            for img in image_files:
                src_file = os.path.join(root, img)
                dst_file = os.path.join(dest_path, img)
                
                if not os.path.exists(dst_file):
                    shutil.copy2(src_file, dst_file)
                    print(f"Copied: {src_file} -> {dst_file}")
                    copied_count += 1
                else:
                    print(f"Skipped (already exists): {dst_file}")
                    skipped_count += 1
    
    print(f"\nSummary: Copied {copied_count} files, Skipped {skipped_count} existing files")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy images containing 'crop' in their name while preserving folder structure")
    parser.add_argument("source", help="Source directory to search for images")
    parser.add_argument("destination", help="Destination directory to copy images to")
    
    args = parser.parse_args()
    copy_crop_images(args.source, args.destination)