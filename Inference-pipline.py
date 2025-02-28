import os
import glob
import tkinter as tk
import threading
import requests
import numpy as np
import torch
from tkinter import filedialog, ttk, messagebox
from PIL import Image
import torch.nn as nn
import torchvision.transforms as transforms

# Define color palette
DARK_PURPLE = "#160c28"
GOLD = "#efcb68"
MINT = "#e1efe6"
GRAY = "#aeb7b3"
DARK_BLUE = "#000411"

# BaseDetector implementation
class BaseDetector(nn.Module):
    """
    Base detector class. This class provides utility methods for
    loading the model, generating results, and performing single and batch image detections.
    """
    
    # Placeholder class-level attributes to be defined in derived classes
    IMAGE_SIZE = None
    STRIDE = None
    CLASS_NAMES = None
    TRANSFORM = None

    def __init__(self, weights=None, device="cpu", url=None):
        """
        Initialize the base detector.
        
        Args:
            weights (str, optional): 
                Path to the model weights. Defaults to None.
            device (str, optional): 
                Device for model inference. Defaults to "cpu".
            url (str, optional): 
                URL to fetch the model weights. Defaults to None.
        """
        super(BaseDetector, self).__init__()
        self.device = device
        self.load_model(weights, device, url)
        
    def load_model(self, weights=None, device="cpu", url=None):
        """
        Load model weights.
        
        Args:
            weights (str, optional): 
                Path to the model weights. Defaults to None.
            device (str, optional): 
                Device for model inference. Defaults to "cpu".
            url (str, optional): 
                URL to fetch the model weights. Defaults to None.
        Raises:
            Exception: If weights are not provided.
        """
        pass
    
    def results_generation(self, preds, img_id, id_strip=None):
        """
        Generate results for detection based on model predictions.
        
        Args:
            preds (numpy.ndarray): 
                Model predictions.
            img_id (str): 
                Image identifier.
            id_strip (str, optional): 
                Strip specific characters from img_id. Defaults to None.
        Returns:
            dict: Dictionary containing image ID, detections, and labels.
        """
        pass
    
    def single_image_detection(self, img, img_size=None, img_path=None, conf_thres=0.2, id_strip=None):
        """
        Perform detection on a single image.
        
        Args:
            img (str or ndarray): 
                Image path or ndarray of images.
            img_size (tuple): 
                Original image size.
            img_path (str): 
                Image path or identifier.
            conf_thres (float, optional): 
                Confidence threshold for predictions. Defaults to 0.2.
            id_strip (str, optional): 
                Characters to strip from img_id. Defaults to None.
        Returns:
            dict: Detection results.
        """
        pass
    
    def batch_image_detection(self, dataloader, conf_thres=0.2, id_strip=None):
        """
        Perform detection on a batch of images.
        
        Args:
            dataloader (DataLoader): 
                DataLoader containing image batches.
            conf_thres (float, optional): 
                Confidence threshold for predictions. Defaults to 0.2.
            id_strip (str, optional): 
                Characters to strip from img_id. Defaults to None.
        Returns:
            list: List of detection results for all images.
        """
        pass


# MegaDetector implementation using the BaseDetector class
class MegaDetector(BaseDetector):
    # Class constants
    IMAGE_SIZE = 640
    STRIDE = 32
    CLASS_NAMES = ['animal', 'person', 'vehicle']
    
    def __init__(self, weights=None, device="cpu", url=None):
        self.TRANSFORM = transforms.Compose([
            transforms.Resize((self.IMAGE_SIZE, self.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        super(MegaDetector, self).__init__(weights, device, url)
    
    def load_model(self, weights=None, device="cpu", url=None):
        """Load MegaDetector model"""
        if weights is None and url is None:
            raise ValueError("Either weights or url must be provided")
        
        try:
            # Load the model
            if weights and os.path.exists(weights):
                print(f"Loading model from {weights}")
                checkpoint = torch.load(weights, map_location=device)
                
                # Handle different model formats
                if 'model' in checkpoint:
                    self.model = checkpoint['model']
                elif 'ema' in checkpoint:
                    self.model = checkpoint['ema']
                else:
                    # Try to load as a state dict
                    from models.yolo import Model  # You might need to adjust this import
                    self.model = Model()
                    self.model.load_state_dict(checkpoint)
                
                self.model = self.model.to(device)
                self.model.eval()
                print("Model loaded successfully")
            else:
                raise FileNotFoundError(f"Model file not found: {weights}")
                
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
    
    def preprocess_image(self, img):
        """Preprocess image for inference"""
        if isinstance(img, str):
            img = Image.open(img).convert('RGB')
        
        # Apply transformations
        img_tensor = self.TRANSFORM(img).unsqueeze(0).to(self.device)
        return img_tensor
    
    def single_image_detection(self, img, img_size=None, img_path=None, conf_thres=0.2, id_strip=None):
        """Detect objects in a single image"""
        try:
            # Preprocess image
            img_tensor = self.preprocess_image(img)
            
            # Run inference
            with torch.no_grad():
                preds = self.model(img_tensor)
            
            # Generate results
            img_id = os.path.basename(img_path) if img_path else "image"
            results = self.results_generation(preds, img_id, id_strip)
            
            # Filter results by confidence
            if 'detections' in results:
                results['detections'] = [
                    det for det in results['detections'] 
                    if det['conf'] >= conf_thres
                ]
            
            return results
        
        except Exception as e:
            print(f"Error in detection: {str(e)}")
            return {"file": img_path, "error": str(e)}
    
    def results_generation(self, preds, img_id, id_strip=None):
        """Generate detection results"""
        if id_strip and img_id.startswith(id_strip):
            img_id = img_id[len(id_strip):]
        
        # Process predictions
        detections = []
        
        # This will need to be adjusted based on the exact format of your model's output
        if isinstance(preds, tuple) and len(preds) > 0:
            # Assume first element contains bounding boxes, confidence scores, and class predictions
            boxes, scores, classes = preds[0], preds[1], preds[2]
            
            for i in range(len(boxes)):
                if scores[i] > 0:
                    x1, y1, x2, y2 = boxes[i]
                    class_id = int(classes[i])
                    if class_id < len(self.CLASS_NAMES):
                        category = self.CLASS_NAMES[class_id]
                    else:
                        category = f"class_{class_id}"
                    
                    detection = {
                        'category': category,
                        'conf': float(scores[i]),
                        'bbox': [float(x1), float(y1), float(x2-x1), float(y2-y1)]
                    }
                    detections.append(detection)
        
        return {
            'file': img_id,
            'detections': detections
        }


# GUI Application class
class ImageDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MegaDetector Image Detection Tool")
        self.root.geometry("800x600")
        self.root.configure(bg=DARK_PURPLE)
        
        # Initialize model variables
        self.model = None
        self.model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "MegaDetectorV6.pt")
        
        # Configure styles
        self.configure_styles()
        
        # Create main frame
        self.main_frame = ttk.Frame(root, style="Main.TFrame")
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Create header
        header_label = ttk.Label(
            self.main_frame, 
            text="MegaDetector Image Detection Tool", 
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
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            self.main_frame, 
            variable=self.progress_var,
            style="TProgressbar",
            mode="determinate"
        )
        
        # Store image paths
        self.image_paths = []
        
        # Check if model needs to be downloaded
        self.check_model()
    
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
    
    def check_model(self):
        """Check if model exists and download if necessary"""
        if not os.path.exists(self.model_path):
            download_thread = threading.Thread(target=self.download_model)
            download_thread.start()
            
            # Show progress bar
            self.progress_bar.pack(pady=10, fill=tk.X)
            self.status_var.set("Downloading MegaDetector model...")
        else:
            # Model exists, load it
            self.load_model()
    
    def download_model(self):
        """Download the MegaDetector model from Hugging Face"""
        try:
            model_url = "https://huggingface.co/nkarthikeyan/MegaDetectorV6/resolve/main/MegaDetectorV6.pt"
            
            # Stream the response to show progress
            response = requests.get(model_url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
            # Download with progress updates
            downloaded = 0
            with open(self.model_path, 'wb') as f:
                for data in response.iter_content(chunk_size=4096):
                    f.write(data)
                    downloaded += len(data)
                    # Update progress bar
                    if total_size:
                        progress = (downloaded / total_size) * 100
                        self.progress_var.set(progress)
                        self.root.update_idletasks()
            
            # Load model after download
            self.root.after(0, lambda: self.load_model())
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Download Error", f"Failed to download model: {str(e)}"))
            self.status_var.set("Error downloading model")
        finally:
            # Hide progress bar
            self.root.after(0, lambda: self.progress_bar.pack_forget())
    
    def load_model(self):
        """Load the MegaDetector model"""
        try:
            self.status_var.set("Loading MegaDetector model...")
            
            # Load model in a separate thread to prevent UI freezing
            def load_model_thread():
                try:
                    # Set device
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    self.model = MegaDetector(weights=self.model_path, device=device)
                    self.root.after(0, lambda: self.status_var.set("Model loaded successfully. Ready for detection."))
                except Exception as inner_e:
                    self.root.after(0, lambda: messagebox.showerror(
                        "Model Loading Error", 
                        f"Failed to load model: {str(inner_e)}"
                    ))
                    self.root.after(0, lambda: self.status_var.set("Error loading model"))
            
            threading.Thread(target=load_model_thread).start()
            
        except Exception as e:
            messagebox.showerror("Model Loading Error", f"Failed to load model: {str(e)}")
            self.status_var.set("Error loading model")
    
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
        if not self.image_paths or self.model is None:
            if self.model is None:
                messagebox.showinfo("Model Not Ready", "Please wait for the model to finish loading.")
            return
        
        self.status_var.set("Running detection...")
        self.detect_btn.config(state=tk.DISABLED)
        self.results_text.delete(1.0, tk.END)
        self.progress_bar.pack(pady=10, fill=tk.X)
        self.progress_var.set(0)
        
        # Run detection in a separate thread to prevent UI freezing
        detection_thread = threading.Thread(target=self.run_detection_thread)
        detection_thread.start()
    
    def run_detection_thread(self):
        """Run detection in a separate thread to prevent UI freezing"""
        try:
            # Process each image
            for i, img_path in enumerate(self.image_paths):
                # Update progress
                progress = (i / len(self.image_paths)) * 100
                self.progress_var.set(progress)
                
                # Update text area
                self.root.after(0, lambda path=img_path: self.results_text.insert(tk.END, f"Processing {os.path.basename(path)}...\n"))
                self.root.update_idletasks()
                
                # Run detection
                results = self.model.single_image_detection(img_path, img_path=img_path, conf_thres=0.2)
                
                # Format and display results
                self.root.after(0, lambda res=results: self.format_and_display_results(res))
            
            # Update UI when done
            self.root.after(0, lambda: self.progress_bar.pack_forget())
            self.root.after(0, lambda: self.status_var.set(f"Detection completed on {len(self.image_paths)} images"))
            self.root.after(0, lambda: self.detect_btn.config(state=tk.NORMAL))
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Detection Error", f"Error during detection: {str(e)}"))
            self.root.after(0, lambda: self.status_var.set("Error during detection"))
            self.root.after(0, lambda: self.detect_btn.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.progress_bar.pack_forget())
    
    def format_and_display_results(self, results):
        """Format and display detection results"""
        if 'error' in results:
            self.results_text.insert(tk.END, f"Error: {results['error']}\n\n")
            return
        
        if 'detections' in results and len(results['detections']) > 0:
            self.results_text.insert(tk.END, "Detected objects:\n")
            for det in results['detections']:
                category = det['category'].capitalize()
                conf = det['conf'] * 100
                self.results_text.insert(tk.END, f"  - {category} (Confidence: {conf:.1f}%)\n")
        else:
            self.results_text.insert(tk.END, "No objects detected.\n")
        
        self.results_text.insert(tk.END, "\n")

def main():
    root = tk.Tk()
    app = ImageDetectionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()