import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import cv2
import shutil
from datetime import datetime

def test_model_inference(model_path, test_image_dir=None, img_size=(224, 224), class_names=None):
    """
    Tests if a model can be loaded and used for inference on images containing 'crop' in their filename.
    Images with predictions are saved to a new 'inference results' folder.
    Excludes images located in folders named "empty".
    
    Args:
        model_path: Path to the .h5 model file
        test_image_dir: Path to directory containing test images
        img_size: Image dimensions expected by the model (height, width)
        class_names: List of class names corresponding to model outputs
    """
    # Normalize path
    model_path = os.path.normpath(model_path)
    
    # 1. Check if model file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return False
    
    try:
        # 2. Try to load the model
        print(f"Loading model from {model_path}...")
        model = load_model(model_path)
        print("Model loaded successfully!")
        
        # 3. Print model summary
        print("\nModel Summary:")
        model.summary()
        
        # 4. Create output directory for results
        if test_image_dir and os.path.exists(test_image_dir):
            output_dir = os.path.join(os.path.dirname(test_image_dir), "inference results v13")
            os.makedirs(output_dir, exist_ok=True)
            print(f"\nCreated output directory: {output_dir}")
            
            # Find all images with 'crop' in the filename
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']
            crop_images = []
            
            for ext in image_extensions:
                pattern = os.path.join(test_image_dir, f"**/*crop*{ext}")
                found_images = glob.glob(pattern, recursive=True)
                
                # Filter out images from folders named "empty"
                for img_path in found_images:
                    # Check if "empty" folder is in the path
                    path_parts = os.path.normpath(img_path).split(os.sep)
                    if "empty" not in path_parts:
                        crop_images.append(img_path)
            
            if not crop_images:
                print(f"No valid images with 'crop' in filename found in {test_image_dir}")
                return False
            
            # Print statistics
            all_crop_images_count = sum(len(glob.glob(os.path.join(test_image_dir, f"**/*crop*{ext}"), recursive=True)) for ext in image_extensions)
            excluded_count = all_crop_images_count - len(crop_images)
            
            print(f"Found {len(crop_images)} valid images with 'crop' in filename")
            print(f"Excluded {excluded_count} images from folders named 'empty'")
            
            # Process each image
            for img_path in crop_images:
                print(f"\nProcessing image: {os.path.basename(img_path)}")
                
                # Load and preprocess the image
                img = load_img(img_path, target_size=img_size, color_mode = 'grayscale')
                img_array = img_to_array(img)
                img_array_normalized = img_array / 255.0  # Normalize to [0,1]
                img_batch = np.expand_dims(img_array_normalized, axis=0)  # Add batch dimension
                
                # Run inference
                predictions = model.predict(img_batch, verbose=0)
                
                # Get top prediction
                predicted_class = np.argmax(predictions[0])
                confidence = predictions[0][predicted_class]
                
                # Get class name if available
                if class_names and predicted_class < len(class_names):
                    class_name = class_names[predicted_class]
                else:
                    class_name = f"Class {predicted_class}"
                
                # Get top 3 predictions
                top_indices = np.argsort(predictions[0])[-3:][::-1]
                top_predictions = []
                for i, idx in enumerate(top_indices):
                    if class_names and idx < len(class_names):
                        cls_name = class_names[idx]
                    else:
                        cls_name = f"Class {idx}"
                    top_predictions.append(f"{cls_name}: {predictions[0][idx]:.2f}")
                
                # Load the image for overlay
                img_cv = cv2.imread(img_path)
                if img_cv is None:
                    print(f"Warning: Could not read image {img_path} with OpenCV, skipping overlay")
                    continue
                
                # Get image dimensions
                img_height, img_width = img_cv.shape[:2]
                
                # Check if image is low resolution (less than 200x200)
                is_low_res = img_width < 200 or img_height < 200
                
                if is_low_res:
                    # For low-resolution images, create a canvas with extra space below the image
                    # Calculate the space needed for text
                    text_height = 40 * len(top_predictions)  # Roughly 40 pixels per line
                    canvas = np.zeros((img_height + text_height, max(img_width, 300), 3), dtype=np.uint8)
                    
                    # Place the original image at the top
                    canvas[:img_height, :img_width] = img_cv
                    
                    # Add prediction text below the image
                    text_position_y = img_height + 30
                    for i, pred_text in enumerate(top_predictions):
                        # Determine text color (red for top prediction, white for others)
                        color = (0, 0, 255) if i == 0 else (255, 255, 255)
                        
                        # Add prediction text
                        cv2.putText(canvas, 
                                   pred_text, 
                                   (10, text_position_y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 
                                   0.7, color, 2)
                        text_position_y += 30
                    
                    img_overlay = canvas
                else:
                    # For higher resolution images, use the original approach
                    img_overlay = img_cv.copy()
                    
                    # Add text with prediction information
                    text_position_y = 30
                    for i, pred_text in enumerate(top_predictions):
                        # Determine text color (red for top prediction, white for others)
                        color = (0, 0, 255) if i == 0 else (255, 255, 255)
                        
                        # Add rectangle behind text for better visibility
                        text_size = cv2.getTextSize(pred_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                        cv2.rectangle(img_overlay, 
                                     (10, text_position_y - 20), 
                                     (10 + text_size[0], text_position_y + 5),
                                     (0, 0, 0), -1)
                        
                        # Add prediction text
                        cv2.putText(img_overlay, 
                                   pred_text, 
                                   (10, text_position_y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 
                                   0.7, color, 2)
                        text_position_y += 30
                
                # Save the output image
                output_filename = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(img_path))[0]}_inference.jpg")
                cv2.imwrite(output_filename, img_overlay)
                print(f"Saved prediction overlay to: {output_filename}")
            
            print(f"\nProcessed {len(crop_images)} images. Results saved to {output_dir}")
            return True
        else:
            # No valid image directory provided
            print(f"Error: Test image directory not found or not provided: {test_image_dir}")
            return False
        
    except Exception as e:
        print(f"Error during model testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Replace with your actual model path
    model_path = "C:/Users/c0062193/OneDrive - Newcastle University/General - Fox-AI/AI results/Model performance/fox_v14_synthetic_20250425_151559/models/best_model_fox_v14_synthetic_20250425_151559.h5"
    
    # Path to the directory containing test images
    test_image_dir = "D:/Perdix-images/ActivityMedia/SITE/99953/"
    
    # Class names for your model
    CLASS_NAMES = ['fox', 'lagomorph', 'person', 'squirrel', 'badger', 'dog', 'bird', 'deer', 'muntjack', 'boar']
    
    # Run the test
    success = test_model_inference(model_path, test_image_dir, class_names=CLASS_NAMES)
    
    if success:
        print("\nModel inference completed successfully! Images with predictions are saved in the 'inference results' folder.")
    else:
        print("\nModel inference failed. Please check the error messages above.")