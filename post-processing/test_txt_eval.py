
# =============================================================================
# ALTERNATIVE USAGE EXAMPLES
# =============================================================================

"""
# Example 1: Basic usage with default save directory
classes = ['fox', 'person', 'badger', 'deer', 'bird', 'squirrel', 'lagomorph']
evaluator = CNNEvaluator('/path/to/model.h5', classes)
results = evaluator.evaluate(['/path/to/file1.txt', '/path/to/file2.txt'])

# Example 2: Custom save directory
evaluator = CNNEvaluator('/path/to/model.h5', classes, save_dir='/custom/save/path')
results = evaluator.evaluate(['/path/to/file1.txt'])

# Example 3: Multiple evaluation runs
evaluator = CNNEvaluator('/path/to/model.h5', classes, save_dir='./results/run1')
train_results = evaluator.evaluate(['/path/to/train.txt'])

evaluator = CNNEvaluator('/path/to/model.h5', classes, save_dir='./results/run2') 
test_results = evaluator.evaluate(['/path/to/test.txt'])
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, confusion_matrix, f1_score, precision_score, recall_score
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import os
from pathlib import Path

class CNNEvaluator:
    def __init__(self, model_path, classes, save_dir):
        """
        Initialize the CNN evaluator
        
        Args:
            model_path (str): Path to the trained Keras model
            classes (list): List of class names
            save_dir (str): Directory to save figures and results
        """
        self.model = keras.models.load_model(model_path)
        self.classes = classes
        self.n_classes = len(classes)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        print(f"Figures will be saved to: {self.save_dir.absolute()}")
        
    def load_data_from_txt(self, txt_file_path):
        """
        Load image paths and labels from txt file
    
        Args:
            txt_file_path (str): Path to txt file with format "image_path<whitespace>class"
        
        Returns:
            tuple: (image_paths, labels)
        """
        image_paths = []
        labels = []
    
        with open(txt_file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    # Debug: Show what we're trying to split
                    if line_num <= 3:  # Show first 3 lines for debugging
                        print(f"Line {line_num}: '{line}'")
                        print(f"  Raw bytes: {repr(line)}")
                
                    # Try different splitting approaches
                    # First try splitting by any whitespace (spaces, tabs, etc.)
                    parts = line.split()
                
                    if len(parts) >= 2:
                        # Take everything except the last part as the path
                        # and the last part as the class name
                        img_path = ' '.join(parts[:-1])
                        class_name = parts[-1]
                    
                        if line_num <= 3:  # Debug output
                            print(f"  Split into: path='{img_path}', class='{class_name}'")
                    
                        if class_name in self.classes:
                            image_paths.append(img_path.strip())
                            labels.append(self.classes.index(class_name.strip()))
                        else:
                            print(f"Warning: Unknown class '{class_name}' in line {line_num}: {line}")
                    else:
                        print(f"Warning: Invalid line format at line {line_num}: {line}")
                        print(f"  Split result: {parts}")
    
        print(f"Successfully parsed {len(image_paths)} valid entries from {txt_file_path}")
        return image_paths, np.array(labels)
    
    def preprocess_image(self, img_path, target_size=(224, 224)):
        """
        Preprocess a single image for prediction
        
        Args:
            img_path (str): Path to image
            target_size (tuple): Target size for resizing
            
        Returns:
            np.array: Preprocessed image
        """
        try:
            img = image.load_img(img_path, target_size=target_size, interpolation = 'lanczos', color_mode = 'grayscale')
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0  # Normalize to [0,1]
            return img_array
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
            return None
    
    def predict_images(self, image_paths):
        """
        Get predictions for a list of images
        
        Args:
            image_paths (list): List of image paths
            
        Returns:
            np.array: Prediction probabilities
        """
        predictions = []
        valid_indices = []
        
        for i, img_path in enumerate(image_paths):
            img_array = self.preprocess_image(img_path)
            if img_array is not None:
                pred = self.model.predict(img_array, verbose=0)
                predictions.append(pred[0])
                valid_indices.append(i)
            else:
                print(f"Skipping invalid image: {img_path}")
        
        return np.array(predictions), valid_indices
    
    def plot_precision_recall_curves(self, y_true, y_pred_proba):
        """
        Plot three figures:
        1. Precision vs Confidence for fox class
        2. Recall vs Confidence for fox class  
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
        # Check if fox class exists
        if 'fox' not in self.classes:
            print("Fox class not found in classes list")
            return
    
        fox_idx = self.classes.index('fox')
        y_true_fox = (y_true == fox_idx).astype(int)
        y_pred_fox = y_pred_proba[:, fox_idx]
    
        # Generate confidence thresholds from 0 to 1
        confidence_thresholds = np.linspace(0, 1, 101)
    
        precisions = []
        recalls = []
    
        for threshold in confidence_thresholds:
            # Get predictions above threshold
            above_threshold = y_pred_fox >= threshold
        
            if np.sum(above_threshold) == 0:
                # No predictions above threshold
                precisions.append(np.nan)
                recalls.append(0)
            else:
                # Calculate precision and recall for predictions above threshold
                y_pred_binary = above_threshold.astype(int)
            
                # Precision: TP / (TP + FP) among predictions above threshold
                tp = np.sum((y_true_fox == 1) & (y_pred_binary == 1))
                fp = np.sum((y_true_fox == 0) & (y_pred_binary == 1))
            
                if tp + fp > 0:
                    precision = tp / (tp + fp)
                else:
                    precision = np.nan
            
                # Recall: TP / (TP + FN) - fraction of actual foxes detected
                fn = np.sum((y_true_fox == 1) & (y_pred_binary == 0))
                if tp + fn > 0:
                    recall = tp / (tp + fn)
                else:
                    recall = np.nan
            
                precisions.append(precision)
                recalls.append(recall)
                
        # Convert to numpy arrays for easier handling
        precisions = np.array(precisions)
        recalls = np.array(recalls)
    
        # Plot 1: Precision vs Confidence
        valid_precision = ~np.isnan(precisions)
        ax1.plot(confidence_thresholds[valid_precision], precisions[valid_precision], 
                 linewidth=2, color='blue', marker='o', markersize=2)
        ax1.set_xlabel('Confidence Threshold')
        ax1.set_ylabel('Precision')
        ax1.set_title('Precision vs Confidence - Fox Class')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
    
        # Plot 2: Recall vs Confidence
        valid_recall = ~np.isnan(recalls)
        ax2.plot(confidence_thresholds[valid_recall], recalls[valid_recall], 
                 linewidth=2, color='red', marker='o', markersize=2)
        ax2.set_xlabel('Confidence Threshold')
        ax2.set_ylabel('Recall')
        ax2.set_title('Recall vs Confidence - Fox Class')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
    
        plt.tight_layout()
        save_path = self.save_dir / "precision_recall_confidence_analysis.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved precision/recall/confidence analysis to: {save_path}")
        plt.show()
    
    def plot_confusion_matrices(self, y_true, y_pred_proba, confidences=[0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]):
        """
        Plot normalized and absolute confusion matrices at different confidence thresholds
        """
        n_conf = len(confidences)
        # Fixed: Create 2 columns and n_conf rows
        fig, axes = plt.subplots(n_conf, 2, figsize=(15, n_conf*5))
    
        # Handle case where there's only one confidence level
        if n_conf == 1:
            axes = axes.reshape(1, -1)
    
        for i, conf in enumerate(confidences):
            # Get predictions at this confidence threshold
            y_pred = np.argmax(y_pred_proba, axis=1)
            max_probs = np.max(y_pred_proba, axis=1)
        
            # Only keep predictions above confidence threshold
            confident_mask = max_probs >= conf
            y_true_conf = y_true[confident_mask]
            y_pred_conf = y_pred[confident_mask]
        
            if len(y_true_conf) == 0:
                print(f"No predictions above confidence {conf}")
                # Clear the axes for this row if no data
                axes[i, 0].text(0.5, 0.5, f'No data\n(conf={conf})', 
                               ha='center', va='center', transform=axes[i, 0].transAxes)
                axes[i, 0].set_xticks([])
                axes[i, 0].set_yticks([])
                axes[i, 1].text(0.5, 0.5, f'No data\n(conf={conf})', 
                               ha='center', va='center', transform=axes[i, 1].transAxes)
                axes[i, 1].set_xticks([])
                axes[i, 1].set_yticks([])
                continue
        
            # Compute confusion matrices
            cm_abs = confusion_matrix(y_true_conf, y_pred_conf, labels=range(self.n_classes))
            cm_norm = confusion_matrix(y_true_conf, y_pred_conf, labels=range(self.n_classes), normalize='true')
        
            # Plot absolute confusion matrix (left column)
            sns.heatmap(cm_abs, annot=True, fmt='d', cmap='magma', 
                       xticklabels=self.classes, yticklabels=self.classes, ax=axes[i, 0])
            axes[i, 0].set_title(f'Absolute CM (conf={conf})\nSamples: {len(y_true_conf)}')
            axes[i, 0].set_xlabel('Predicted')
            axes[i, 0].set_ylabel('True')
        
            # Plot normalized confusion matrix (right column)
            sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='magma', 
                       xticklabels=self.classes, yticklabels=self.classes, ax=axes[i, 1])
            axes[i, 1].set_title(f'Normalized CM (conf={conf})\nSamples: {len(y_true_conf)}')
            axes[i, 1].set_xlabel('Predicted')
            axes[i, 1].set_ylabel('True')
    
        plt.tight_layout()
        save_path = self.save_dir / "confusion_matrices.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved confusion matrices to: {save_path}")
        plt.show()
    
    def plot_f1_curves(self, y_true, y_pred_proba):
        """
        Plot F1 curves for fox class and all classes, showing optimal thresholds
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Fox class F1 curve
        if 'fox' in self.classes:
            fox_idx = self.classes.index('fox')
            y_true_fox = (y_true == fox_idx).astype(int)
            y_pred_fox = y_pred_proba[:, fox_idx]
            
            thresholds = np.linspace(0, 1, 100)
            f1_scores_fox = []
            
            for thresh in thresholds:
                y_pred_thresh = (y_pred_fox >= thresh).astype(int)
                if np.sum(y_pred_thresh) > 0 and np.sum(y_true_fox) > 0:
                    f1 = f1_score(y_true_fox, y_pred_thresh)
                else:
                    f1 = 0
                f1_scores_fox.append(f1)
            
            f1_scores_fox = np.array(f1_scores_fox)
            best_thresh_fox = thresholds[np.argmax(f1_scores_fox)]
            best_f1_fox = np.max(f1_scores_fox)
            
            ax1.plot(thresholds, f1_scores_fox, linewidth=2, label='Fox F1 Score')
            ax1.axvline(best_thresh_fox, color='red', linestyle='--', 
                       label=f'Optimal: {best_thresh_fox:.3f} (F1={best_f1_fox:.3f})')
            ax1.set_xlabel('Confidence Threshold')
            ax1.set_ylabel('F1 Score')
            ax1.set_title('F1 Score vs Confidence - Fox Class')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
        
        # All classes (macro-averaged F1)
        thresholds = np.linspace(0, 1, 100)
        f1_scores_macro = []
        f1_scores_classes = {class_name: [] for class_name in self.classes}
        
        for thresh in thresholds:
            class_f1s = []
            
            for class_idx, class_name in enumerate(self.classes):
                y_true_class = (y_true == class_idx).astype(int)
                y_pred_class = (y_pred_proba[:, class_idx] >= thresh).astype(int)
                
                if np.sum(y_pred_class) > 0 and np.sum(y_true_class) > 0:
                    f1 = f1_score(y_true_class, y_pred_class)
                else:
                    f1 = 0
                
                class_f1s.append(f1)
                f1_scores_classes[class_name].append(f1)
            
            f1_scores_macro.append(np.mean(class_f1s))
        
        # Plot macro-averaged F1
        f1_scores_macro = np.array(f1_scores_macro)
        best_thresh_macro = thresholds[np.argmax(f1_scores_macro)]
        best_f1_macro = np.max(f1_scores_macro)
        
        ax2.plot(thresholds, f1_scores_macro, linewidth=3, label='Macro-averaged F1', color='black')
        ax2.axvline(best_thresh_macro, color='red', linestyle='--', 
                   label=f'Optimal: {best_thresh_macro:.3f} (F1={best_f1_macro:.3f})')
        
        # Plot individual class F1 curves
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.classes)))
        for i, (class_name, f1_scores) in enumerate(f1_scores_classes.items()):
            ax2.plot(thresholds, f1_scores, alpha=0.7, color=colors[i], label=class_name)
        
        ax2.set_xlabel('Confidence Threshold')
        ax2.set_ylabel('F1 Score')
        ax2.set_title('F1 Score vs Confidence - All Classes')
        ax2.grid(True, alpha=0.3)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        save_path = self.save_dir / "f1_curves.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved F1 curves to: {save_path}")
        plt.show()
        
        return {
            'fox_optimal_threshold': best_thresh_fox if 'fox' in self.classes else None,
            'fox_optimal_f1': best_f1_fox if 'fox' in self.classes else None,
            'macro_optimal_threshold': best_thresh_macro,
            'macro_optimal_f1': best_f1_macro
        }
    
    def evaluate(self, txt_files):
        """
        Complete evaluation pipeline
        
        Args:
            txt_files (list): List of paths to txt files containing image paths and labels
        """
        # Load all data
        all_image_paths = []
        all_labels = []
        
        for txt_file in txt_files:
            print(f"Loading data from {txt_file}...")
            image_paths, labels = self.load_data_from_txt(txt_file)
            all_image_paths.extend(image_paths)
            all_labels.extend(labels)
        
        all_labels = np.array(all_labels)
        print(f"Loaded {len(all_image_paths)} images with {len(np.unique(all_labels))} unique classes")
        
        # Get predictions
        print("Getting model predictions...")
        y_pred_proba, valid_indices = self.predict_images(all_image_paths)
        y_true = all_labels[valid_indices]
        
        print(f"Successfully processed {len(y_pred_proba)} images")
        
        # Generate all plots and metrics
        print("Generating precision-recall curves...")
        self.plot_precision_recall_curves(y_true, y_pred_proba)
        
        print("Generating confusion matrices...")
        self.plot_confusion_matrices(y_true, y_pred_proba)
        
        print("Generating F1 curves and finding optimal thresholds...")
        optimal_thresholds = self.plot_f1_curves(y_true, y_pred_proba)
        
        # Print summary
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        
        if optimal_thresholds['fox_optimal_threshold'] is not None:
            print(f"Fox class optimal threshold: {optimal_thresholds['fox_optimal_threshold']:.3f}")
            print(f"Fox class best F1 score: {optimal_thresholds['fox_optimal_f1']:.3f}")
        
        print(f"All classes optimal threshold: {optimal_thresholds['macro_optimal_threshold']:.3f}")
        print(f"All classes best F1 score: {optimal_thresholds['macro_optimal_f1']:.3f}")
        
        # Class distribution
        print(f"\nClass distribution in dataset:")
        unique, counts = np.unique(y_true, return_counts=True)
        for class_idx, count in zip(unique, counts):
            print(f"  {self.classes[class_idx]}: {count} samples")
        
        print(f"\nAll figures saved to: {self.save_dir.absolute()}")
        
        return optimal_thresholds

# EXAMPLE USAGE - SCRIPT CONFIGURATION
if __name__ == "__main__":
    # =============================================================================
    # CONFIGURATION - MODIFY THESE PATHS FOR YOUR SETUP
    # =============================================================================
    
    # Specify your model path
    MODEL_PATH = 'C:/Users/c0062193.CAMPUS/OneDrive - Newcastle University/General - Fox-AI/AI results/Model performance/fox_v23_cleanednewdata_20250627_172030/models/best_model_fox_v23_cleanednewdata_20250627_172030.h5' 
    #fox_v20_cleaned_20250603_121146 #FoxSpot_v20 #best_model_fox_v21_newdata_20250618_185001.h5' #best_model_fox_v22_newdata_20250625_172502 #fox_v23_cleanednewdata_20250627_172030
    
    # Specify your txt files with image paths and labels
    TXT_FILES = [
        'G:/Data/data_v16/test_set_paths_data_v16.txt',
        'G:/Data/data_v15/Removed/test_set_paths_removed.txt'
    ]
    
    # Specify where to save the figures
    SAVE_DIRECTORY = 'C:/Users/c0062193.CAMPUS/OneDrive - Newcastle University/General - Fox-AI/AI results/Model performance/fox_v23_cleanednewdata_20250627_172030/tests_v16/'
    os.makedirs(SAVE_DIRECTORY,exist_ok=True)
    
    # Define classes (these should match your model's classes)
    CLASSES = ['fox', 'person', 'badger', 'deer', 'bird', 'squirrel', 'lagomorph']
    
    # =============================================================================
    # RUN EVALUATION
    # =============================================================================
    
    print("Starting CNN Model Evaluation...")
    print(f"Model: {MODEL_PATH}")
    print(f"Data files: {TXT_FILES}")
    print(f"Save directory: {SAVE_DIRECTORY}")
    print(f"Classes: {CLASSES}")
    print("-" * 50)
    
    # Initialize evaluator with save directory
    evaluator = CNNEvaluator(MODEL_PATH, CLASSES, SAVE_DIRECTORY)
    
    # Run complete evaluation
    results = evaluator.evaluate(TXT_FILES)
    
    print("\nEvaluation completed successfully!")
    print(f"Results: {results}")
