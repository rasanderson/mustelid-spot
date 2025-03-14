import os
import glob

def rename_animal_to_fox(folder_path):
    # Count files processed and renamed
    total_files = 0
    renamed_files = 0
    
    # Get all files in the specified folder and its subfolders
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            total_files += 1
            file_path = os.path.join(root, filename)
            
            # Check if "animal" is in the filename
            if "roe deer" in filename:
                # Create the new filename with "animal" replaced by "fox"
                new_filename = filename.replace("roe deer", "muntjack")
                new_file_path = os.path.join(root, new_filename)
                
                # Rename the file
                try:
                    os.rename(file_path, new_file_path)
                    renamed_files += 1
                    print(f"Renamed: {filename} -> {new_filename}")
                except Exception as e:
                    print(f"Error renaming {filename}: {e}")
    
    print(f"\nSummary: Checked {total_files} files, renamed {renamed_files} files.")

# Use the specific folder path
folder_path = r"C:\Users\c0062193.CAMPUS\OneDrive - Newcastle University\General - Fox-AI\Processed\Not Fox\preprocessed\preprocessed_munjack_deer_images"
rename_animal_to_fox(folder_path)