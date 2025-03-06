import tensorflow as tf
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PytorchWildlife.models import detection as pw_detection


# Set your paths here
images_folder = r"C:/Users/c0062193.CAMPUS/OneDrive - Newcastle University/PhD/Jobs/Fox detection/Images/"
target_size = (224, 224)  # Set to match your model's input size
class_names = ['fox', 'lagomorph', 'person', 'squirrel', 'badger', 'dog']  # Set your class names here
model_base_path = r"C:/Users/c0062193.CAMPUS/OneDrive - Newcastle University/General - Fox-AI/AI results/Model performance/fox_v7_20250305_115154/models"
model_h5_path = os.path.join(model_base_path, 'best_model_fox_v7_20250305_115154.h5')   
model_path = model_h5_path

detection_model = pw_detection.MegaDetectorV6()

def main():
    # Load model
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Find images (including in subdirectories)
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        image_files.extend(glob.glob(os.path.join(images_folder, f'**/*{ext}'), recursive=True))
        image_files.extend(glob.glob(os.path.join(images_folder, f'**/*{ext.upper()}'), recursive=True))
    
    if not image_files:
        print(f"No images found in {images_folder}")
        return
    
    print(f"Found {len(image_files)} images")
    
    # Create figure for results
    n_images = min(len(image_files), 16)  # Display at most 16 images
    cols = min(4, n_images)
    rows = (n_images + cols - 1) // cols
    
    plt.figure(figsize=(16, 4 * rows))
    
    # Process each image
    results = []
    for i, img_path in enumerate(image_files[:n_images]):
        # Load and preprocess image
        img = load_img(img_path, target_size=target_size, color_mode='grayscale')
        img_array = img_to_array(img)
        img_array = img_array / 255.0  # Normalize to [0,1]
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        
        # Make prediction
        pred = model.predict(img_array, verbose=0)
        
        # Determine prediction class and score
        if pred.shape[1] > 1:  # Multi-class
            pred_class = np.argmax(pred[0])
            pred_score = pred[0][pred_class]
        else:  # Binary
            pred_score = pred[0][0]
            pred_class = 1 if pred_score > 0.5 else 0
        
        # Get class name
        if class_names and pred_class < len(class_names):
            pred_label = class_names[pred_class]
        else:
            pred_label = f"Class {pred_class}"
        
        # Store result
        results.append({
            'filename': os.path.basename(img_path),
            'class': pred_label,
            'score': float(pred_score)
        })
        
        # Plot image
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img)
        plt.title(f"{pred_label} ({pred_score:.2f})")
        plt.axis('off')
    
    plt.tight_layout()
    
    # Save results
    output_dir = os.path.dirname(model_path)
    plt.savefig(os.path.join(output_dir, 'predictions.png'))
    
    # Print results
    print("\nPrediction Results:")
    for result in results:
        print(f"Image: {result['filename']}, Prediction: {result['class']} ({result['score']:.4f})")
    
    # Show plot
    plt.show()
    
if __name__ == "__main__":
    main()