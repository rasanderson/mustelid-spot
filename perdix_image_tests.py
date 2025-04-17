import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shutil
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import f1_score, precision_recall_curve, roc_curve, auc
from tqdm import tqdm

def evaluate_cnn_model(model_path, test_dir, output_dir, img_size=(224, 224), grayscale=False):
    """
    Evaluates a CNN model on test images and organizes results.
    
    Args:
        model_path: path to the h5 model file
        test_dir: directory containing class folders with test images
        output_dir: directory to save results and organized images
        img_size: tuple of (height, width) for image resizing
        grayscale: whether to load images as grayscale (single channel)
    """
    # 1. Load the model
    print("Loading model...")
    model = tf.keras.models.load_model(model_path)
    
    # Get class labels from folder names
    class_labels = sorted([d for d in os.listdir(test_dir) 
                          if os.path.isdir(os.path.join(test_dir, d))])
    
    print(f"Found {len(class_labels)} classes: {class_labels}")
    
    # 2. Create output directories
    correct_dir = os.path.join(output_dir, "correct_classifications")
    incorrect_dir = os.path.join(output_dir, "incorrect_classifications")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(correct_dir, exist_ok=True)
    os.makedirs(incorrect_dir, exist_ok=True)
    
    # Create class subdirectories in correct and incorrect folders
    for class_name in class_labels:
        os.makedirs(os.path.join(correct_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(incorrect_dir, class_name), exist_ok=True)
    
    # 3. Process test images
    all_filenames = []
    true_labels = []
    pred_probs = []
    
    print("Processing test images...")
    
    for class_idx, class_name in enumerate(class_labels):
        class_dir = os.path.join(test_dir, class_name)
        image_files = [f for f in os.listdir(class_dir) 
                       if "crop" in f.lower() and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"Processing {len(image_files)} images from class '{class_name}'...")
        
        for img_file in tqdm(image_files):
            img_path = os.path.join(class_dir, img_file)
            
            # Load and preprocess image
            color_mode = 'grayscale' if grayscale else 'rgb'
            img = load_img(img_path, target_size=img_size, color_mode=color_mode)
            img_array = img_to_array(img)
            img_array = img_array / 255.0  # Normalize to [0,1]
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
            
            # Get model predictions
            predictions = model.predict(img_array, verbose=0)[0]
            pred_class_idx = np.argmax(predictions)
            pred_class_name = class_labels[pred_class_idx]
            confidence = predictions[pred_class_idx]
            
            # Store results
            all_filenames.append(img_path)
            true_labels.append(class_idx)
            pred_probs.append(predictions)
            
            # Organize images into correct/incorrect folders
            dest_dir = correct_dir if pred_class_name == class_name else incorrect_dir
            dest_path = os.path.join(dest_dir, class_name, img_file)
            shutil.copy2(img_path, dest_path)
    
    # Convert to numpy arrays for analysis
    true_labels = np.array(true_labels)
    pred_probs = np.array(pred_probs)
    pred_labels = np.argmax(pred_probs, axis=1)
    
    # 4. Calculate metrics and create plots
    print("Generating performance metrics and plots...")
    
    # Prepare for multi-class metrics
    from sklearn.preprocessing import label_binarize
    n_classes = len(class_labels)
    y_true_bin = label_binarize(true_labels, classes=range(n_classes))
    
    # 4.1 F1 score by confidence threshold plot
    plt.figure(figsize=(12, 6))
    thresholds = np.linspace(0.1, 1.0, 19)  # [0.1, 0.15, ..., 0.95, 1.0]
    f1_scores = []
    
    for threshold in thresholds:
        # Apply threshold to predictions
        thresholded_preds = np.zeros_like(pred_labels)
        for i, probs in enumerate(pred_probs):
            if np.max(probs) >= threshold:
                thresholded_preds[i] = np.argmax(probs)
            else:
                # Assign most frequent class when below threshold
                thresholded_preds[i] = np.argmax(np.bincount(true_labels))
        
        f1 = f1_score(true_labels, thresholded_preds, average='weighted')
        f1_scores.append(f1)
    
    # Find optimal threshold
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    optimal_f1 = f1_scores[optimal_idx]
    
    plt.plot(thresholds, f1_scores, marker='o')
    plt.axvline(x=optimal_threshold, color='r', linestyle='--', 
                label=f'Optimal threshold: {optimal_threshold:.2f}, F1: {optimal_f1:.3f}')
    plt.title('F1 Score by Confidence Threshold')
    plt.xlabel('Confidence Threshold')
    plt.ylabel('Weighted F1 Score')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'f1_by_threshold.png'), dpi=300)
    
    # 4.2 True/False Positive Rate Curve (ROC curve)
    plt.figure(figsize=(12, 8))
    
    # Calculate ROC curve and AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], pred_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Calculate micro-average ROC curve and AUC
    fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), pred_probs.ravel())
    roc_auc_micro = auc(fpr_micro, tpr_micro)
    
    # Plot ROC curves
    plt.plot(fpr_micro, tpr_micro, 'b-', label=f'Micro-average ROC (AUC = {roc_auc_micro:.3f})')
    
    # Plot ROC for each class
    colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, n_classes))
    for i, color in zip(range(n_classes), colors):
        class_name = class_labels[i] if i < len(class_labels) else f"Class {i}"
        plt.plot(fpr[i], tpr[i], color=color, lw=1.5,
                 label=f'ROC for {class_name} (AUC = {roc_auc[i]:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=1.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right", fontsize='small')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curves.png'), dpi=300)
    
    # Generate summary report
    print("\nModel Evaluation Summary:")
    print(f"Total test images: {len(all_filenames)}")
    
    # Count correct/incorrect predictions
    correct = sum(true_labels == pred_labels)
    incorrect = len(true_labels) - correct
    accuracy = correct / len(true_labels) * 100
    
    print(f"Correct classifications: {correct} ({accuracy:.2f}%)")
    print(f"Incorrect classifications: {incorrect} ({100-accuracy:.2f}%)")
    print(f"Optimal confidence threshold: {optimal_threshold:.2f} (F1: {optimal_f1:.3f})")
    
    # Generate per-class metrics
    class_report = pd.DataFrame({
        'Class': class_labels,
        'Images': np.bincount(true_labels),
        'Correct': [sum((true_labels == i) & (pred_labels == i)) for i in range(n_classes)],
        'AUC': [roc_auc[i] for i in range(n_classes)]
    })
    
    class_report['Accuracy'] = class_report['Correct'] / class_report['Images'] * 100
    
    print("\nPer-class performance:")
    print(class_report)
    
    # Save results to CSV
    class_report.to_csv(os.path.join(output_dir, 'class_performance.csv'), index=False)
    
    # Create a DataFrame with detailed results for each image
    results_df = pd.DataFrame({
        'Filename': all_filenames,
        'True_Class': [class_labels[i] for i in true_labels],
        'Predicted_Class': [class_labels[i] for i in pred_labels],
        'Confidence': [pred_probs[i, pred_labels[i]] for i in range(len(pred_labels))],
        'Correct': true_labels == pred_labels
    })
    
    results_df.to_csv(os.path.join(output_dir, 'detailed_results.csv'), index=False)
    
    print(f"\nResults saved to: {output_dir}")
    print(f"Images organized in: {correct_dir} and {incorrect_dir}")

# Example usage
if __name__ == "__main__":
    # Replace these paths with your actual paths
    model_path = "C:/Users/c0062193.CAMPUS/OneDrive - Newcastle University/General - Fox-AI/AI results/Model performance/fox_v10_20250325_170446/models/best_model_fox_v10_20250325_170446.h5"
    test_dir = "C:/Users/c0062193.CAMPUS/OneDrive - Newcastle University/PhD/Jobs/Fox detection/Results/perdix_test/test_images/"  # Should contain subdirectories for each class
    output_dir = "C:/Users/c0062193.CAMPUS/OneDrive - Newcastle University/PhD/Jobs/Fox detection/Results/perdix_test/Output_v10"
    
    # Optional: Specify image size based on your model's input requirements
    img_size = (224, 224)  # (height, width)
    
    # Set to True if your model expects grayscale images
    grayscale = True
    
    evaluate_cnn_model(model_path, test_dir, output_dir, img_size, grayscale)