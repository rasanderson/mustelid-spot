import os
import glob
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk  # Required for image handling

# Define color palette
DARK_PURPLE = "#160c28"
GOLD = "#efcb68"
MINT = "#e1efe6"
GRAY = "#aeb7b3"
DARK_BLUE = "#000411"

class ImageDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Detection Tool")
        self.root.geometry("800x600")
        self.root.configure(bg=DARK_PURPLE)
        
        # Configure styles
        self.configure_styles()
        
        # Create main frame
        self.main_frame = ttk.Frame(root, style="Main.TFrame")
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Create header
        header_label = ttk.Label(
            self.main_frame, 
            text="Image Detection Tool", 
            style="Header.TLabel"
        )
        header_label.pack(pady=(0, 20))
        
        # Create folder selection button
        self.select_folder_btn = ttk.Button(
            self.main_frame,
            text="Select Image Folder",
            command=self.select_folder,
            style="Gold.TButton"
        )
        self.select_folder_btn.pack(pady=10)
        
        # Label to display selected folder path
        self.folder_path_var = tk.StringVar()
        self.folder_path_var.set("No folder selected")
        folder_path_label = ttk.Label(
            self.main_frame, 
            textvariable=self.folder_path_var,
            style="Path.TLabel"
        )
        folder_path_label.pack(pady=10)
        
        # Create detection button
        self.detect_btn = ttk.Button(
            self.main_frame,
            text="Run Detection",
            command=self.run_detection,
            style="Blue.TButton"
        )
        self.detect_btn.pack(pady=10)
        self.detect_btn.config(state=tk.DISABLED)
        
        # Create results area
        results_frame = ttk.Frame(self.main_frame, style="Results.TFrame")
        results_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Scrollable results text area
        self.results_text = tk.Text(
            results_frame,
            bg=MINT,
            fg=DARK_BLUE,
            font=("Arial", 11),
            wrap=tk.WORD,
            height=15
        )
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Add scrollbar to results text
        scrollbar = ttk.Scrollbar(results_frame, command=self.results_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_text.config(yscrollcommand=scrollbar.set)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(
            root, 
            textvariable=self.status_var,
            style="Status.TLabel"
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Store image paths
        self.image_paths = []
    
    def configure_styles(self):
        """Configure ttk styles with the color palette"""
        style = ttk.Style()
        
        # Frame styles
        style.configure("Main.TFrame", background=DARK_PURPLE)
        style.configure("Results.TFrame", background=MINT)
        
        # Label styles
        style.configure("Header.TLabel", 
                       background=DARK_PURPLE, 
                       foreground=GOLD,
                       font=("Arial", 16, "bold"))
        style.configure("Path.TLabel", 
                       background=DARK_PURPLE, 
                       foreground=MINT,
                       font=("Arial", 10))
        style.configure("Status.TLabel", 
                       background=DARK_BLUE, 
                       foreground=MINT,
                       font=("Arial", 9))
        
        # Button styles
        style.configure("Gold.TButton", 
                       background=GOLD, 
                       foreground=DARK_PURPLE,
                       font=("Arial", 11, "bold"))
        style.configure("Blue.TButton", 
                       background=DARK_BLUE, 
                       foreground=MINT,
                       font=("Arial", 11, "bold"))
    
    def select_folder(self):
        """Open a dialog to select a folder containing images"""
        folder_path = filedialog.askdirectory(title="Select Folder Containing Images")
        if folder_path:
            self.folder_path_var.set(folder_path)
            self.status_var.set(f"Selected folder: {folder_path}")
            self.image_paths = self.glob_images(folder_path)
            
            if self.image_paths:
                self.results_text.delete(1.0, tk.END)
                self.results_text.insert(tk.END, f"Found {len(self.image_paths)} images:\n\n")
                for i, path in enumerate(self.image_paths[:10], 1):  # Show first 10 images
                    self.results_text.insert(tk.END, f"{i}. {os.path.basename(path)}\n")
                
                if len(self.image_paths) > 10:
                    self.results_text.insert(tk.END, f"\n... and {len(self.image_paths) - 10} more\n")
                
                self.detect_btn.config(state=tk.NORMAL)
            else:
                self.results_text.delete(1.0, tk.END)
                self.results_text.insert(tk.END, "No images found in the selected folder.")
                self.detect_btn.config(state=tk.DISABLED)
    
    def glob_images(self, folder_path):
        """Find all image files in the specified folder"""
        # Common image extensions
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif', '*.tiff']
        image_paths = []
        
        for ext in extensions:
            pattern = os.path.join(folder_path, '**', ext)
            image_paths.extend(glob.glob(pattern, recursive=True))
        
        return image_paths
    
    def run_detection(self):
        """Run detection on the images"""
        if not self.image_paths:
            return
        
        self.status_var.set("Running detection...")
        self.detect_btn.config(state=tk.DISABLED)
        self.results_text.delete(1.0, tk.END)
        
        # Process each image
        for i, img_path in enumerate(self.image_paths):
            self.results_text.insert(tk.END, f"Processing {os.path.basename(img_path)}...\n")
            
            # Update the UI
            self.root.update_idletasks()
            
            # This is where you would integrate your detection model
            detection_results = self.detect_objects(img_path)
            
            # Display results for this image
            self.results_text.insert(tk.END, f"Results: {detection_results}\n\n")
        
        self.status_var.set(f"Detection completed on {len(self.image_paths)} images")
        self.detect_btn.config(state=tk.NORMAL)
    
    def detect_objects(self, image_path):
        """
        Run object detection on a single image
        Replace this with your actual detection model implementation
        """
        # PLACEHOLDER: Replace with your actual detection model
        # For example, you might use:
        # - TensorFlow/Keras models
        # - PyTorch models
        # - OpenCV-based detection
        # - A pre-trained model from Hugging Face
        
        # This is just a dummy implementation
        return "Placeholder detection results. Replace with actual model."

def main():
    root = tk.Tk()
    app = ImageDetectionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()