import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def test_model_inference(model_path, test_image_path=None, img_size=(224, 224), class_names=None):
    """
    Tests if a model can be loaded and used for inference on a sample image.
    
    Args:
        model_path: Path to the .h5 model file
        test_image_path: Path to a test image (optional)
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
        
        # 4. Test inference with either a provided image or random data
        if test_image_path and os.path.exists(test_image_path):
            # Load and preprocess a real image
            print(f"\nPerforming inference on image: {test_image_path}")
            img = load_img(test_image_path, target_size=img_size)
            img_array = img_to_array(img)
            img_array = img_array / 255.0  # Normalize to [0,1]
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        else:
            # Generate random input data
            print("\nNo test image provided or found. Using random test data...")
            input_shape = model.input_shape
            # Remove None (batch dimension) from input shape
            if input_shape[0] is None:
                input_shape = input_shape[1:]
            print(f"Using random input with shape: {input_shape}")
            img_array = np.random.random((1,) + input_shape)
        
        # 5. Run inference
        print("Running inference...")
        predictions = model.predict(img_array, verbose=1)
        
        # 6. Print prediction shape and values
        print(f"\nPrediction shape: {predictions.shape}")
        print(f"Prediction values: {predictions[0]}")
        
        if len(predictions[0]) > 1:  # Classification model with multiple classes
            predicted_class = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class]
            print(f"\nPredicted class index: {predicted_class}")
            print(f"Confidence: {confidence:.4f}")
            
            # Print class name if available
            if class_names and predicted_class < len(class_names):
                print(f"Predicted class: {class_names[predicted_class]}")
                
            # Show top 3 predictions with class names
            if class_names:
                top_indices = np.argsort(predictions[0])[-3:][::-1]
                print("\nTop 3 predictions:")
                for i, idx in enumerate(top_indices):
                    class_name = class_names[idx] if idx < len(class_names) else f"Unknown ({idx})"
                    print(f"  {i+1}. {class_name}: {predictions[0][idx]:.4f}")
        
        return True
        
    except Exception as e:
        print(f"Error during model testing: {str(e)}")
        return False

if __name__ == "__main__":
    # Replace with your actual model path
    model_path = "C:/Users/c0062193.CAMPUS/OneDrive - Newcastle University/General - Fox-AI/AI results/Model performance/fox_v11_synthetic_20250416_155355/models/best_model_fox_v11_synthetic_20250416_155355.h5"
    
    # Optional: Path to a test image if available
    test_image_path = "C:/Users/c0062193.CAMPUS/OneDrive - Newcastle University/PhD/Jobs/Fox detection/Results/perdix_test/fox/"  # Replace with a test image path if you have one
    
    # Class names for your model
    CLASS_NAMES = ['fox', 'lagomorph', 'person', 'squirrel', 'badger', 'dog', 'bird', 'deer', 'muntjack', 'boar']
    
    # Run the test
    success = test_model_inference(model_path, test_image_path, class_names=CLASS_NAMES)
    
    if success:
        print("\nModel test completed successfully! The model can be loaded and used for inference.")
    else:
        print("\nModel test failed. Please check the error messages above.")