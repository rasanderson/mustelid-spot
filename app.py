import os
import spaces  # Import spaces for ZeroGPU
import torch
import numpy as np
import gradio as gr
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
import supervision as sv
from PytorchWildlife.models import detection as pw_detection
import PytorchWildlife.utils as pw_utils
import io
import base64
from functools import lru_cache

# ZeroGPU device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Load models - Initialize as None, load in GPU-decorated functions
MODEL_NAME = "FoxSpot_v20"
MODEL_NAME_v22 ="best_model_fox_v22_newdata_20250625_172502"
MODEL_PATH = f"{MODEL_NAME_v22}.h5"

# Global model variables
classification_model = None
detection_model = None

# Configure TensorFlow for better performance
try:
    if tf.config.list_physical_devices('GPU'):
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Configured TensorFlow GPU memory growth")
    else:
        # Optimize for CPU if no GPU available
        tf.config.threading.set_intra_op_parallelism_threads(0)  # Use all available cores
        tf.config.threading.set_inter_op_parallelism_threads(0)  # Use all available cores
        print("Configured TensorFlow for CPU optimization")
except Exception as e:
    print(f"TensorFlow configuration warning: {e}")

def load_models():
    """Load models on GPU when needed - optimized version"""
    global classification_model, detection_model
    
    if classification_model is None:
        try:
            # Load TensorFlow model with optimizations
            classification_model = tf.keras.models.load_model(
                MODEL_PATH,
                compile=False  # Skip compilation for faster loading
            )
            
            # Try to optimize compilation if possible
            #try:
            #    classification_model.compile(
            #        optimizer='adam',  # Use string instead of object for better compatibility
            #        run_eagerly=False
            #    )
            #    print("Optimized classification model compilation")
            #except Exception as comp_e:
            #    print(f"Could not optimize compilation (non-critical): {comp_e}")
            #
            #print(f"Successfully loaded classification model from {MODEL_PATH}")
        except Exception as e:
            print(f"Error loading classification model: {e}")
            classification_model = None
    
    if detection_model is None:
        try:
            # Load PyTorch detection model - use safer version
            detection_model = pw_detection.MegaDetectorV6(
                device=DEVICE, 
                pretrained=True, 
                version="MDV6-yolov10-c"  # Use 'c' version which might be more stable
            )
            
            # Try to optimize if the model attribute exists
            try:
                if hasattr(detection_model, 'model') and detection_model.model is not None:
                    detection_model.model.eval()
                    print("Set detection model to eval mode")
                elif hasattr(detection_model, 'detector') and detection_model.detector is not None:
                    detection_model.detector.eval()
                    print("Set detector to eval mode")
            except Exception as opt_e:
                print(f"Could not optimize detection model (non-critical): {opt_e}")
            
            print("Successfully loaded MegaDetectorV6")
        except Exception as e:
            print(f"Error loading detection model: {e}")
            detection_model = None

# Define class names for your classifier with emojis
CLASS_NAMES = ['fox', 'person', 'badger', 'deer', 'bird', 'squirrel', 'lagomorph']
CLASS_ICONS = {
    'fox': '🦊',
    'person': '👤',
    'badger': '🦡',
    'deer': '🦌',
    'bird': '🐦',
    'squirrel': '🐿️',
    'lagomorph': '🐰'
}

@lru_cache(maxsize=32)
def preprocess_for_classification_cached(image_bytes, target_size=(224, 224)):
    """Cached preprocessing for classification model."""
    # Convert bytes back to PIL Image
    image = Image.open(io.BytesIO(image_bytes))
    
    # Convert to grayscale - ensure it's actually grayscale
    if image.mode != 'L':
        image = image.convert('L')
    
    # Resize
    image = image.resize(target_size, Image.LANCZOS)  # Use better resampling
    
    # Convert to numpy and normalize
    img_array = np.array(image, dtype = "float32") / 255.0  # Use float32 for better performance
    
    # Ensure it's 2D (grayscale), then add channel dimension
    if len(img_array.shape) == 2:
        # Add channel dimension for grayscale (224, 224, 1)
        img_array = np.expand_dims(img_array, axis=-1)
    
    # Add batch dimension (1, 224, 224, 1)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def preprocess_for_classification(image, target_size=(224, 224)):
    """Preprocess cropped image for classification model."""
    # Convert to PIL if numpy array
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Convert to bytes for caching
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    image_bytes = buffer.getvalue()
    
    return preprocess_for_classification_cached(image_bytes, target_size)

def get_class_name(prediction, classification_threshold=0.5):
    """Convert model prediction to class name, applying confidence threshold."""
    class_idx = np.argmax(prediction[0])
    confidence = np.max(prediction[0])
    class_name = CLASS_NAMES[class_idx]
    
    if confidence < classification_threshold:
        return "uncertain", confidence
    
    return class_name, confidence

def draw_detection_boxes(image, detections):
    """Draw simple detection boxes on the original image without classification."""
    # Convert to PIL if numpy array
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    else:
        image = image.copy()
    
    draw = ImageDraw.Draw(image)
    
    # Use fox orange color for all boxes
    color = '#FF6600'  # Fox orange
    
    for i, detection in enumerate(detections):
        bbox = detection['bbox']
        
        # Draw simple rectangle without label
        draw.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], outline=color, width=2)
    
    return image

def create_crop_gallery(cropped_images, classifications):
    """Create vertical stack of cropped animals with labels - optimized."""
    if not cropped_images or not classifications:
        return "<p style='text-align: center; color: #666; padding: 20px;'>🖼️ Detected animal crops will appear here</p>"
    
    gallery_html = """
    <div style="
        display: flex; 
        flex-direction: column; 
        gap: 12px; 
        max-height: 55vh; 
        overflow-y: auto; 
        overflow-x: hidden;
        padding: 10px; 
        width: 100%;
        box-sizing: border-box;
    ">
    """
    
    for i, (crop_img, classification) in enumerate(zip(cropped_images, classifications)):
        if crop_img is None:
            continue
            
        # Convert PIL image to base64 for HTML display - optimize image size
        buffer = io.BytesIO()
        # Resize crop for display to reduce base64 size
        display_crop = crop_img.copy()
        if display_crop.size[0] > 150 or display_crop.size[1] > 150:
            display_crop.thumbnail((150, 150), Image.LANCZOS)
        
        display_crop.save(buffer, format='JPEG', quality=85, optimize=True)
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        # Get classification info
        class_text = classification['classification']
        
        # Extract class name and confidence
        if 'Classification failed' in class_text:
            display_text = "❓ Classification Failed"
            icon = "❓"
        elif 'Low confidence' in class_text or 'uncertain' in class_text.lower():
            confidence = class_text.split('(')[1].split(')')[0] if '(' in class_text else "Low"
            display_text = f"❓ Uncertain ({confidence})"
            icon = "❓"
        else:
            # Parse "ClassName (XX.X%)"
            parts = class_text.split(' (')
            if len(parts) == 2:
                class_name = parts[0]
                confidence = parts[1].replace(')', '')
                icon = CLASS_ICONS.get(class_name, '🔍')
                display_text = f"{icon} {class_name.title()} ({confidence})"
            else:
                display_text = f"🔍 {class_text}"
                icon = "🔍"
        
        gallery_html += f"""
        <div style="
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center; 
            border: 2px solid #FF6600; 
            border-radius: 8px; 
            padding: 12px; 
            background: white; 
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            width: 100%;
            box-sizing: border-box;
            min-height: 160px;
        ">
            <div style="
                font-weight: bold; 
                font-size: 12px; 
                margin-bottom: 8px; 
                color: #FF6600;
            ">
                Detection #{i+1}
            </div>
            <div style="
                display: flex;
                align-items: center;
                justify-content: center;
                width: 100%;
                height: 100px;
                margin-bottom: 8px;
            ">
                <img src="data:image/jpeg;base64,{img_str}" style="
                    max-width: 100%; 
                    max-height: 100%; 
                    width: auto;
                    height: auto;
                    border-radius: 4px;
                    object-fit: contain;
                " />
            </div>
            <div style="
                font-size: 11px; 
                color: #333; 
                font-weight: bold;
                word-wrap: break-word;
                text-align: center;
                width: 100%;
            ">
                {display_text}
            </div>
        </div>
        """
    
    gallery_html += "</div>"
    return gallery_html

def create_summary(animal_detections, classifications):
    """Create summary text."""
    if not animal_detections:
        return "📤 Upload an image and click 'Analyze Image' to start detection and classification"
    
    # Start with detection summary
    summary_lines = [f"🔍 **Detection Summary**"]
    summary_lines.append(f"Found {len(animal_detections)} animals")
    
    # Count species if classifications are available
    if classifications:
        species_count = {}
        for result in classifications:
            class_text = result['classification']
            if 'Classification failed' in class_text:
                species = 'Failed'
            elif 'Low confidence' in class_text or 'uncertain' in class_text.lower():
                species = 'Uncertain'
            else:
                species = class_text.split(' (')[0]
            
            species_count[species] = species_count.get(species, 0) + 1
        
        if species_count:
            summary_lines.append("")
            summary_lines.append("📊 **Species Breakdown**:")
            for species, count in species_count.items():
                icon = CLASS_ICONS.get(species, '❓') if species in CLASS_ICONS else '❓'
                summary_lines.append(f"• {icon} {species.title()}: {count}")
    
    return "\n".join(summary_lines)

# GPU decorator with fallback
def gpu_decorator(func):
    """Decorator that tries to use GPU but falls back gracefully"""
    try:
        return spaces.GPU(func)
    except Exception as e:
        print(f"ZeroGPU decorator failed, using regular function: {e}")
        return func

@gpu_decorator  # Use fallback GPU decorator
def process_image(input_image, detection_threshold, classification_threshold):
    """Main processing pipeline: Detection -> Classification - optimized version"""
    
    # Check if an image was provided
    if input_image is None:
        return ("📤 Upload an image to start detection and classification", 
                None, 
                "<p style='text-align: center; color: #666; padding: 20px;'>🖼️ Detected animal crops will appear here</p>")
    
    # Convert to numpy if needed
    if not isinstance(input_image, np.ndarray):
        input_image = np.array(input_image)
    
    # Resize input image if too large to speed up processing
    max_size = 1024
    if input_image.shape[0] > max_size or input_image.shape[1] > max_size:
        pil_img = Image.fromarray(input_image)
        pil_img.thumbnail((max_size, max_size), Image.LANCZOS)
        input_image = np.array(pil_img)
        print(f"Resized image to {input_image.shape} for faster processing")
    
    try:
        # Ensure models are loaded
        load_models()
        
        if detection_model is None:
            return ("❌ Error: Detection model failed to load. Please check model files.", 
                    None, 
                    "<p style='text-align: center; color: #666;'>Detection model loading failed</p>")
        
        if classification_model is None:
            return ("❌ Error: Classification model failed to load. Please check model files.", 
                    None, 
                    "<p style='text-align: center; color: #666;'>Classification model loading failed</p>")
        
        # PHASE 1: DETECTION
        print("="*50)
        print("STARTING DETECTION PHASE")
        print("="*50)
        
        results_det = detection_model.single_image_detection(
            input_image, 
            det_conf_thres=detection_threshold
        )
        
        # Filter for animal detections only (class_id == 0 for animals in MegaDetector)
        animal_detections = []
        for i, (xyxy, det_id, confidence) in enumerate(zip(
            results_det["detections"].xyxy, 
            results_det["detections"].class_id,
            results_det["detections"].confidence
        )):
            if det_id == 0:  # Animal class
                animal_detections.append({
                    'bbox': xyxy,
                    'confidence': confidence,
                    'detection_id': i
                })
        
        print(f"Found {len(animal_detections)} animal detections")
        
        if len(animal_detections) == 0:
            return ("🚫 No animals detected in the image. Try lowering the detection threshold.", 
                    Image.fromarray(input_image), 
                    "<p style='text-align: center; color: #666;'>No animals detected</p>")
        
        # Show detection boxes immediately (before classification)
        detection_image = draw_detection_boxes(input_image, animal_detections)
        
        # PHASE 2: CLASSIFICATION - Batch process for efficiency
        print("="*50)
        print("STARTING CLASSIFICATION PHASE")
        print("="*50)
        
        classification_results = []
        cropped_images = []
        
        # Prepare all crops first
        crops_to_process = []
        for i, detection in enumerate(animal_detections):
            try:
                # Crop the detected animal
                cropped_animal = sv.crop_image(image=input_image, xyxy=detection['bbox'])
                cropped_images.append(Image.fromarray(cropped_animal))
                
                # Preprocess for classification
                processed_crop = preprocess_for_classification(cropped_animal)
                crops_to_process.append((i, processed_crop, detection))
                
            except Exception as e:
                print(f"Error processing detection {i+1}: {str(e)}")
                cropped_images.append(None)
                classification_results.append({
                    'detection_id': detection['detection_id'],
                    'classification': f"Processing failed: {str(e)}",
                    'detection_confidence': detection['confidence'],
                    'bbox': detection['bbox']
                })
        
        # Batch classify all crops at once for better GPU utilization
        if crops_to_process:
            try:
                # Stack all preprocessed crops into a single batch
                batch_crops = np.vstack([crop[1] for crop in crops_to_process])
                
                # Run batch prediction
                batch_predictions = classification_model.predict(batch_crops, verbose=0)
                
                # Process results
                for (i, _, detection), prediction in zip(crops_to_process, batch_predictions):
                    class_name, confidence = get_class_name(
                        [prediction], classification_threshold
                    )
                    
                    if class_name == "uncertain":
                        class_text = f"Low confidence detection ({confidence*100:.1f}%)"
                    else:
                        class_text = f"{class_name} ({confidence*100:.1f}%)"
                    
                    # Insert result at correct position
                    while len(classification_results) <= i:
                        classification_results.append(None)
                    
                    classification_results[i] = {
                        'detection_id': detection['detection_id'],
                        'classification': class_text,
                        'detection_confidence': detection['confidence'],
                        'bbox': detection['bbox']
                    }
                    
                    print(f"Detection {i+1}: {class_text}")
                    
            except Exception as e:
                print(f"Error in batch classification: {str(e)}")
                # Fallback to individual processing
                for i, processed_crop, detection in crops_to_process:
                    try:
                        prediction = classification_model.predict(processed_crop, verbose=0)
                        class_name, confidence = get_class_name(prediction, classification_threshold)
                        
                        if class_name == "uncertain":
                            class_text = f"Low confidence detection ({confidence*100:.1f}%)"
                        else:
                            class_text = f"{class_name} ({confidence*100:.1f}%)"
                        
                        while len(classification_results) <= i:
                            classification_results.append(None)
                            
                        classification_results[i] = {
                            'detection_id': detection['detection_id'],
                            'classification': class_text,
                            'detection_confidence': detection['confidence'],
                            'bbox': detection['bbox']
                        }
                    except Exception as e2:
                        print(f"Error classifying detection {i+1}: {str(e2)}")
                        while len(classification_results) <= i:
                            classification_results.append(None)
                        classification_results[i] = {
                            'detection_id': detection['detection_id'],
                            'classification': f"Classification failed: {str(e2)}",
                            'detection_confidence': detection['confidence'],
                            'bbox': detection['bbox']
                        }
        
        print("Classification phase complete.")
        
        # Create final outputs
        final_summary = create_summary(animal_detections, classification_results)
        crop_gallery = create_crop_gallery(cropped_images, classification_results)
        
        return final_summary, detection_image, crop_gallery
        
    except Exception as e:
        print(f"Error in processing pipeline: {str(e)}")
        return (f"❌ Error processing image: {str(e)}", 
                None, 
                "<p style='text-align: center; color: #666;'>Error occurred during processing</p>")

# Create Gradio interface using Blocks for improved layout
with gr.Blocks(title="FoxSpot: British Mammals Classifier", css="""
    .gradio-container {
        max-width: 100% !important;
        width: 100% !important;
    }
    .right-column {
        max-height: 70vh;
        overflow-y: auto;
    }
    .main-image {
        max-height: 70vh;
    }
    @media (max-width: 768px) {
        .main-row {
            flex-direction: column !important;
        }
    }
""") as demo:
    gr.Markdown("# 🦊 FoxSpot: British Mammals Classifier")
    gr.Markdown("Upload an image to detect and classify common British mammals.")
    
    # Main layout: Three columns as requested
    with gr.Row():
        # LEFT COLUMN: Upload and controls
        with gr.Column(scale=1, min_width=300):
            input_image = gr.Image(
                label="Upload Image"
            )
            
            detection_slider = gr.Slider(
                minimum=0.1, 
                maximum=1.0, 
                value=0.3,  # Slightly higher default to reduce false positives
                step=0.05, 
                label="Detection Confidence",
                info="Lower = more detections"
            )
            
            classification_slider = gr.Slider(
                minimum=0.1, 
                maximum=1.0, 
                value=0.4,  # Slightly lower default for more classifications
                step=0.05, 
                label="Classification Confidence",
                info="Lower = more classifications"
            )
            
            submit_btn = gr.Button("🔍 Analyze Image", variant="primary", size="lg")
        
        # MIDDLE COLUMN: Image with bounding boxes
        with gr.Column(scale=2, min_width=400):
            annotated_image = gr.Image(
                label="📍 Image with Detection Boxes", 
                interactive=False,
                elem_classes=["main-image"]
            )
        
        # RIGHT COLUMN: Summary and crops stacked vertically
        with gr.Column(scale=1, min_width=250, elem_classes=["right-column"]):
            summary_output = gr.Markdown(
                "📤 Upload an image and click 'Analyze Image' to start detection and classification"
            )
            
            crop_gallery = gr.HTML(
                "<p style='text-align: center; color: #666; padding: 20px;'>🖼️ Detected animal crops will appear here</p>"
            )
    
    # Examples section - moved below main interface to save vertical space
    with gr.Row():
        gr.Examples(
            examples=[
                ["examples/fox-april-2.jpeg"],
                ["examples/intro-badger-1.jpeg"], 
                ["examples/roe-deer.jpeg"],
                ["examples/intro-fox-2.jpeg"]
            ],
            inputs=input_image,
            label="📁 Example Images"
        )
    
    # Set up the event handler
    submit_btn.click(
        fn=process_image,
        inputs=[input_image, detection_slider, classification_slider],
        outputs=[summary_output, annotated_image, crop_gallery]
    )

if __name__ == "__main__":
    # Create examples directory if it doesn't exist
    os.makedirs("examples", exist_ok=True)
    
    # Launch the interface
    demo.launch()