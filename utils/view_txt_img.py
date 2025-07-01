import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from pathlib import Path

DIR = "G:/Data/data_v16"

# Load paths (split on tab to get just the path, ignore the label)
with open(Path(DIR) / "test_set_paths_data_v16.txt", 'r') as f:
    data = []
    for line in f:
        if line.strip():
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                path, label = parts[0], parts[1]
                if os.path.exists(path):
                    data.append((path, label))

print(f"Found {len(data)} valid images")

if not data:
    print("No valid images found!")
    exit()

# Global variables for navigation
current_page = 0
images_per_page = 16
total_pages = (len(data) + images_per_page - 1) // images_per_page

# Create the figure once and reuse it
fig, axes = plt.subplots(4, 4, figsize=(12, 10))
axes = axes.flatten()

def display_page(page_num):
    """Display a page of images"""
    global current_page
    current_page = page_num
    
    # Clear all axes
    for ax in axes:
        ax.clear()
        ax.axis('off')
    
    # Calculate start and end indices
    start_idx = page_num * images_per_page
    end_idx = min(start_idx + images_per_page, len(data))
    page_data = data[start_idx:end_idx]
    
    for i in range(images_per_page):        
        if i < len(page_data):
            path, label = page_data[i]
            try:
                img = mpimg.imread(path)
                
                # Handle grayscale images
                if len(img.shape) == 2:
                    # Grayscale image
                    axes[i].imshow(img, cmap='gray')
                elif len(img.shape) == 3:
                    if img.shape[2] == 1:
                        # Grayscale with single channel
                        axes[i].imshow(img.squeeze(), cmap='gray')
                    else:
                        # Color image
                        axes[i].imshow(img)
                else:
                    raise ValueError("Unexpected image shape")
                
                # Title with filename and label
                filename = os.path.basename(path)
                axes[i].set_title(f"{filename}\n{label}", fontsize=7, pad=1)
                
            except Exception as e:
                axes[i].text(0.5, 0.5, f'Error loading\n{os.path.basename(path)}', 
                       ha='center', va='center', transform=axes[i].transAxes, fontsize=7)
                print(f"Error loading {path}: {e}")
        
        axes[i].axis('off')
    
    # Add page info
    fig.suptitle(f'Page {page_num + 1} of {total_pages} (Images {start_idx + 1}-{end_idx} of {len(data)})', 
                 fontsize=12, y=0.95)
    
    # Adjust layout to fit better
    plt.tight_layout(rect=[0, 0.05, 1, 0.92])
    
    # Show navigation instructions
    fig.text(0.5, 0.02, "Navigation: ← Previous page | → Next page | 'q' to quit", 
             ha='center', fontsize=9, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    
    # Redraw the figure
    fig.canvas.draw()

def on_key(event):
    """Handle key press events"""
    global current_page
    
    if event.key == 'right' or event.key == 'down':
        # Next page
        if current_page < total_pages - 1:
            display_page(current_page + 1)
        else:
            print("Already on last page")
    
    elif event.key == 'left' or event.key == 'up':
        # Previous page
        if current_page > 0:
            display_page(current_page - 1)
        else:
            print("Already on first page")
    
    elif event.key == 'q':
        # Quit
        plt.close('all')
        print("Viewer closed")

# Connect the key press event to the figure
fig.canvas.mpl_connect('key_press_event', on_key)

# Display first page
display_page(0)

print("\nControls:")
print("← ↑ : Previous page")
print("→ ↓ : Next page") 
print("q   : Quit")
print("\nMake sure the plot window is active/focused to use keyboard controls!")

plt.show()