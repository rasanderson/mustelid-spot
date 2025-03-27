import os
import numpy as np
import gradio as gr
import tensorflow as tf
import torch
from PIL import Image
import supervision as sv

# Import PytorchWildlife components
from PytorchWildlife.models import detection as pw_detection
import PytorchWildlife.utils as pw_utils

# Set device (GPU if available)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load models
MODEL_NAME = "fox_v9_20250314_180533"
MODEL_PATH = "models/best_model_{}.h5".format(MODEL_NAME)
classification_model = tf.keras.models.load_model(MODEL_PATH)

# Initialize detector
detection_model = pw_detection.MegaDetectorV5(device=DEVICE)

# Define class names for your classifier
CLASS_NAMES = ['fox', 'lagomorph', 'person', 'squirrel', 'badger', 'dog', 'bird', 'deer', 'muntjack', 'boar']

def preprocess_for_classification(image, target_size=(224, 224)):
    """Preprocess cropped image for classification model."""
    # Convert to PIL if numpy array
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Resize
    image = image.resize(target_size)
    
    # Convert to numpy and normalize
    img_array = np.array(image) / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def get_class_name(prediction):
    """Convert model prediction to class name."""
    class_idx = np.argmax(prediction[0])
    confidence = np.max(prediction[0]) * 100
    return f"{CLASS_NAMES[class_idx]} ({confidence:.1f}%)"

def process_image(input_image):
    """Process image through detection and classification pipeline."""
    # Convert to numpy if needed
    if not isinstance(input_image, np.ndarray):
        input_image = np.array(input_image)
    
    # Step 1: Detect animals using PytorchWildlife's MegaDetector
    results_det = detection_model.single_image_detection(
        input_image, 
        det_conf_thres=0.2
    )
    
    # No detections
    if len(results_det["detections"].xyxy) == 0:
        return "No animals detected", None
    
    # Get the first animal detection
    for xyxy, det_id in zip(results_det["detections"].xyxy, results_det["detections"].class_id):
        # Only process if detection class is animal (class 0 in MegaDetector)
        if det_id == 0:  # 0 is the class ID for 'animal' in MegaDetector
            # Crop the detected animal using supervision library
            cropped_animal = sv.crop_image(image=input_image, xyxy=xyxy)
            
            # Step 2: Classify the cropped animal with your model
            processed_crop = preprocess_for_classification(cropped_animal)
            prediction = classification_model.predict(processed_crop, verbose=0)
            class_name = get_class_name(prediction)
            
            return class_name, Image.fromarray(cropped_animal)
    
    return "Animal detected but classification failed", None

# Create Gradio interface
demo = gr.Interface(
    fn=process_image,
    inputs=gr.Image(),
    outputs=[
        gr.Text(label="Classification"), 
        gr.Image(label="Detected Animal")
    ],
    title="VulpesVision: British Mammals Classifier",
    description="Upload an image to detect and classify British mammals, focusing on foxes and invasive species.",
    allow_flagging="never"
)

if __name__ == "__main__":
    # Launch the interface
    demo.launch(share=True)  # share=True creates a public URL