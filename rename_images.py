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

def rename_child_human(folder_path):
    # Count files processed and renamed
    total_files = 0
    renamed_files = 0
    
    # Get all files in the specified folder and its subfolders
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            total_files += 1
            file_path = os.path.join(root, filename)
            
            # Check if "animal" is in the filename
            if "child" in filename:
                # Create the new filename with "animal" replaced by "fox"
                new_filename = filename.replace("child", "human_child")
                new_file_path = os.path.join(root, new_filename)
                
                # Rename the file
                try:
                    os.rename(file_path, new_file_path)
                    renamed_files += 1
                    print(f"Renamed: {filename} -> {new_filename}")
                except Exception as e:
                    print(f"Error renaming {filename}: {e}")
    
    print(f"\nSummary: Checked {total_files} files, renamed {renamed_files} files.")

def rename_png_files(folder_path):
    import os
    
    # Count files processed and renamed
    total_files = 0
    renamed_files = 0
    
    # Get folder name for the new filenames
    folder_name = os.path.basename(os.path.normpath(folder_path))
    
    # Process only files in the specified folder (not subfolders)
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    
    # Filter for PNG files only
    png_files = [f for f in files if f.lower().endswith('.png')]
    
    # Rename PNG files with sequential numbering
    for index, filename in enumerate(png_files, 1):
        total_files += 1
        file_path = os.path.join(folder_path, filename)
        
        # Create the new filename using folder name and sequential number
        new_filename = f"{folder_name}_{index}.png"
        new_file_path = os.path.join(folder_path, new_filename)
        
        # Rename the file
        try:
            os.rename(file_path, new_file_path)
            renamed_files += 1
            print(f"Renamed: {filename} -> {new_filename}")
        except Exception as e:
            print(f"Error renaming {filename}: {e}")
    
    print(f"\nSummary: Found {total_files} PNG files, renamed {renamed_files} files.")

# Use the specific folder path
folder_path = r"F:\Fox-AI\To process\Flux\bird"
#rename_animal_to_fox(folder_path)
rename_png_files(folder_path)