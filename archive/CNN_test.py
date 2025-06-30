import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import tensorflow as tf
import random
from collections import defaultdict
import json

# Check GPU availability
print("GPU availability check:")
print(f"TensorFlow version: {tf.__version__}")
print(f"GPUs available: {len(tf.config.list_physical_devices('GPU'))}")
if tf.config.list_physical_devices('GPU'):
    for gpu in tf.config.list_physical_devices('GPU'):
        print(f"  {gpu}")
    print("CUDA will be used for predictions")
else:
    print("No GPU detected - using CPU")
print()

class CNNTester:
    def __init__(self, model_path, training_dir, removed_dir, test_dir=None, img_size=(224, 224)):
        """
        Initialize the CNN tester
        
        Args:
            model_path: Path to saved Keras model
            training_dir: Directory with training dataset structure
            removed_dir: Directory with removed images (same structure as training)
            test_dir: Optional directory with dedicated test images (class_name subfolders)
            img_size: Target image size for model input
        """
        self.model = load_model(model_path)
        self.training_dir = training_dir
        self.removed_dir = removed_dir
        self.test_dir = test_dir
        self.img_size = img_size
        # Use predefined class order to match model training
        self.class_names = ['fox', 'person', 'badger', 'deer', 'bird', 'squirrel', 'lagomorph']
        
        # Calculate removal ratios for each class
        self.removal_ratios = self._calculate_removal_ratios()
        print("Removal ratios by class:")
        for class_name, ratio in self.removal_ratios.items():
            print(f"  {class_name}: {ratio:.2%}")
    
    def _get_image_files(self, directory, class_name, is_test_dir=False):
        """Get all image files for a class, handling nested directory structures"""
        images = []
        
        if is_test_dir:
            # Test directory structure: test_dir/class_name/
            # Only select images with "crop" in the filename
            class_path = os.path.join(directory, class_name)
            if os.path.exists(class_path):
                for root, dirs, files in os.walk(class_path):
                    for file in files:
                        if (file.lower().endswith(('.png', '.jpg', '.jpeg')) and 
                            'crop' in file.lower()):
                            images.append(os.path.join(root, file))
        elif class_name == 'fox':
            # Fox structure: Fox/ (with location subfolders)
            fox_path = os.path.join(directory, 'Fox')
            if os.path.exists(fox_path):
                for root, dirs, files in os.walk(fox_path):
                    for file in files:
                        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            images.append(os.path.join(root, file))
        else:
            # Other classes structure: 
            # Both training and removed use: Not Fox/preprocessed_class_name_images/
            not_fox_path = os.path.join(directory, 'Not Fox')
            folder_name = f"preprocessed_{class_name}_images"
            class_path = os.path.join(not_fox_path, folder_name)
            
            if os.path.exists(class_path):
                for root, dirs, files in os.walk(class_path):
                    for file in files:
                        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            images.append(os.path.join(root, file))
            else:
                print(f"  Warning: Path not found for {class_name}: {class_path}")
        
        return images
    
    def _calculate_removal_ratios(self):
        """Calculate what percentage was removed for each class"""
        ratios = {}
        
        for class_name in self.class_names:
            training_images = self._get_image_files(self.training_dir, class_name)
            removed_images = self._get_image_files(self.removed_dir, class_name)
            
            training_count = len(training_images)
            removed_count = len(removed_images)
            
            total_original = training_count + removed_count
            ratios[class_name] = removed_count / total_original if total_original > 0 else 0
            
        return ratios
    
    def create_test_set(self, test_size_per_class=100):
        """
        Create test set with priority: test_dir images > removed images to match removal ratio
        
        Args:
            test_size_per_class: Target number of test images per class
        """
        test_data = []
        
        for class_name in self.class_names:
            print(f"\nProcessing class: {class_name}")
            
            # Get all available images from different sources
            test_images = []
            if self.test_dir:
                test_images = self._get_image_files(self.test_dir, class_name, is_test_dir=True)
            
            removed_images = self._get_image_files(self.removed_dir, class_name)
            
            removal_ratio = self.removal_ratios[class_name]
            
            print(f"  Found {len(test_images)} dedicated test images with 'crop', {len(removed_images)} removed images")
            print(f"  Removal ratio: {removal_ratio:.2%}")
            
            # Calculate how many from each source
            if removal_ratio == 0:
                # No removed images, use all from test set up to test_size_per_class
                test_count = min(len(test_images), test_size_per_class)
                removed_count = 0
            else:
                # Calculate based on removal ratio
                removed_count = int(test_size_per_class * removal_ratio)
                test_count = test_size_per_class - removed_count
                
                # Adjust if we don't have enough images
                test_count = min(test_count, len(test_images))
                removed_count = min(removed_count, len(removed_images))
                
                # If we still don't have enough, fill from the other source
                total_available = test_count + removed_count
                if total_available < test_size_per_class:
                    if len(test_images) > test_count:
                        additional_test = min(len(test_images) - test_count, 
                                            test_size_per_class - total_available)
                        test_count += additional_test
                    elif len(removed_images) > removed_count:
                        additional_removed = min(len(removed_images) - removed_count,
                                               test_size_per_class - total_available)
                        removed_count += additional_removed
            
            print(f"  Selecting: {test_count} dedicated test images, {removed_count} removed images")
            
            # Sample images
            selected_test = random.sample(test_images, test_count) if test_count > 0 else []
            selected_removed = random.sample(removed_images, removed_count) if removed_count > 0 else []
            
            # Add to test data
            for img_path in selected_test:
                test_data.append({
                    'image_path': img_path,
                    'true_class': class_name,
                    'source': 'test_set'
                })
            
            for img_path in selected_removed:
                test_data.append({
                    'image_path': img_path,
                    'true_class': class_name,
                    'source': 'removed'
                })
        
        return test_data
    
    def preprocess_image(self, image_path):
        """Preprocess a single grayscale image for model prediction"""
        try:
            # Load as grayscale
            img = load_img(image_path, target_size=self.img_size, color_mode='grayscale')
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0  # Normalize to [0,1]
            return img_array
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None
    
    def predict_with_confidence(self, test_data):
        """Make predictions and return results with confidence scores"""
        results = []
        
        print(f"\nMaking predictions on {len(test_data)} test images...")
        
        for i, data in enumerate(test_data):
            if i % 50 == 0:
                print(f"  Progress: {i}/{len(test_data)}")
            
            img_array = self.preprocess_image(data['image_path'])
            if img_array is None:
                continue
            
            # Get prediction probabilities
            predictions = self.model.predict(img_array, verbose=0)
            confidence_scores = predictions[0]
            predicted_class_idx = np.argmax(confidence_scores)
            predicted_class = self.class_names[predicted_class_idx]
            max_confidence = confidence_scores[predicted_class_idx]
            
            results.append({
                'image_path': data['image_path'],
                'true_class': data['true_class'],
                'predicted_class': predicted_class,
                'confidence': max_confidence,
                'source': data['source'],
                'all_confidences': confidence_scores.tolist()
            })
        
        return results
    
    def analyze_confidence_thresholds(self, results, thresholds=None):
        """Analyze performance at different confidence thresholds"""
        if thresholds is None:
            thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
        
        analysis_results = {}
        
        for threshold in thresholds:
            # Filter predictions by confidence threshold
            confident_predictions = [r for r in results if r['confidence'] >= threshold]
            
            if len(confident_predictions) == 0:
                continue
            
            # Calculate metrics
            y_true = [r['true_class'] for r in confident_predictions]
            y_pred = [r['predicted_class'] for r in confident_predictions]
            
            # Overall accuracy
            accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)
            
            # Class-specific metrics
            class_metrics = {}
            for class_name in self.class_names:
                class_true = [r for r in confident_predictions if r['true_class'] == class_name]
                class_correct = [r for r in class_true if r['predicted_class'] == class_name]
                
                # True positives, false positives, false negatives
                tp = len(class_correct)
                fn = len(class_true) - tp
                fp = len([r for r in confident_predictions 
                         if r['predicted_class'] == class_name and r['true_class'] != class_name])
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                class_metrics[class_name] = {
                    'tp': tp, 'fp': fp, 'fn': fn,
                    'precision': precision, 'recall': recall, 'f1': f1,
                    'samples': len(class_true)
                }
            
            analysis_results[threshold] = {
                'accuracy': accuracy,
                'total_samples': len(confident_predictions),
                'coverage': len(confident_predictions) / len(results),
                'class_metrics': class_metrics
            }
        
        return analysis_results
    
    def find_optimal_thresholds(self, analysis_results):
        """Find optimal confidence threshold for each class based on F1 score"""
        optimal_thresholds = {}
        
        for class_name in self.class_names:
            best_f1 = 0
            best_threshold = 0.5
            
            for threshold, metrics in analysis_results.items():
                if class_name in metrics['class_metrics']:
                    f1 = metrics['class_metrics'][class_name]['f1']
                    if f1 > best_f1:
                        best_f1 = f1
                        best_threshold = threshold
            
            optimal_thresholds[class_name] = {
                'threshold': best_threshold,
                'f1_score': best_f1
            }
        
        return optimal_thresholds
    
    def analyze_errors(self, results, threshold=0.5):
        """Detailed analysis of false positives and false negatives"""
        # Filter by confidence threshold
        filtered_results = [r for r in results if r['confidence'] >= threshold]
        
        false_positives = defaultdict(list)
        false_negatives = defaultdict(list)
        
        for result in filtered_results:
            true_class = result['true_class']
            pred_class = result['predicted_class']
            
            if true_class != pred_class:
                # False negative for true class
                false_negatives[true_class].append(result)
                # False positive for predicted class
                false_positives[pred_class].append(result)
        
        return false_positives, false_negatives
    
    def create_false_positive_visualization(self, results, output_dir, threshold=0.5):
        """Create visualization showing false positive patterns"""
        # Get false positives
        false_positives, _ = self.analyze_errors(results, threshold)
        
        # Create matrix showing what each class gets confused as
        confusion_data = np.zeros((len(self.class_names), len(self.class_names)))
        
        filtered_results = [r for r in results if r['confidence'] >= threshold]
        for result in filtered_results:
            if result['true_class'] != result['predicted_class']:
                true_idx = self.class_names.index(result['true_class'])
                pred_idx = self.class_names.index(result['predicted_class'])
                confusion_data[true_idx, pred_idx] += 1
        
        # Create the visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # 1. False positive counts by predicted class
        fp_counts = {class_name: len(fps) for class_name, fps in false_positives.items()}
        classes = list(fp_counts.keys())
        counts = list(fp_counts.values())
        
        bars = ax1.bar(classes, counts, color='coral', alpha=0.7)
        ax1.set_title(f'False Positives by Predicted Class\n(Confidence ≥ {threshold})', fontsize=14)
        ax1.set_xlabel('Predicted Class', fontsize=12)
        ax1.set_ylabel('Number of False Positives', fontsize=12)
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{int(height)}', ha='center', va='bottom', fontsize=10)
        
        # 2. Confusion heatmap (what gets confused as what)
        im = ax2.imshow(confusion_data, cmap='Reds', interpolation='nearest')
        ax2.set_title(f'Confusion Pattern Matrix\n(Confidence ≥ {threshold})', fontsize=14)
        ax2.set_xlabel('Predicted Class', fontsize=12)
        ax2.set_ylabel('True Class', fontsize=12)
        
        # Set ticks and labels
        ax2.set_xticks(range(len(self.class_names)))
        ax2.set_yticks(range(len(self.class_names)))
        ax2.set_xticklabels(self.class_names, rotation=45, ha='right')
        ax2.set_yticklabels(self.class_names)
        
        # Add text annotations
        for i in range(len(self.class_names)):
            for j in range(len(self.class_names)):
                if confusion_data[i, j] > 0:
                    text = ax2.text(j, i, int(confusion_data[i, j]),
                                   ha="center", va="center", color="black", fontsize=10)
        
        # Add colorbar
        plt.colorbar(im, ax=ax2, shrink=0.8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'false_positives_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create detailed breakdown by source
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Count false positives by source and predicted class
        fp_by_source = defaultdict(lambda: defaultdict(int))
        for class_name, fps in false_positives.items():
            for fp in fps:
                fp_by_source[fp['source']][class_name] += 1
        
        # Prepare data for stacked bar chart
        sources = list(fp_by_source.keys())
        class_counts = {class_name: [] for class_name in self.class_names}
        
        for source in sources:
            for class_name in self.class_names:
                class_counts[class_name].append(fp_by_source[source][class_name])
        
        # Create stacked bar chart
        bottom = np.zeros(len(sources))
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.class_names)))
        
        for i, class_name in enumerate(self.class_names):
            if sum(class_counts[class_name]) > 0:  # Only plot if there are false positives
                ax.bar(sources, class_counts[class_name], bottom=bottom, 
                      label=class_name, color=colors[i], alpha=0.8)
                bottom += class_counts[class_name]
        
        ax.set_title(f'False Positives by Source and Predicted Class\n(Confidence ≥ {threshold})', fontsize=14)
        ax.set_xlabel('Image Source', fontsize=12)
        ax.set_ylabel('Number of False Positives', fontsize=12)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'false_positives_by_source.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"False positive visualizations saved to {output_dir}/")
    
    def create_visualizations(self, results, analysis_results, output_dir):
        """Create comprehensive visualizations"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Confidence threshold analysis
        thresholds = list(analysis_results.keys())
        accuracies = [analysis_results[t]['accuracy'] for t in thresholds]
        coverages = [analysis_results[t]['coverage'] for t in thresholds]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        ax1.plot(thresholds, accuracies, 'bo-', label='Accuracy')
        ax1.set_xlabel('Confidence Threshold')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Accuracy vs Confidence Threshold')
        ax1.grid(True)
        ax1.legend()
        
        ax2.plot(thresholds, coverages, 'ro-', label='Coverage')
        ax2.set_xlabel('Confidence Threshold')
        ax2.set_ylabel('Coverage (% of samples)')
        ax2.set_title('Coverage vs Confidence Threshold')
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'threshold_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Per-class F1 scores across thresholds
        plt.figure(figsize=(12, 8))
        for class_name in self.class_names:
            f1_scores = []
            valid_thresholds = []
            for t in thresholds:
                if class_name in analysis_results[t]['class_metrics']:
                    f1_scores.append(analysis_results[t]['class_metrics'][class_name]['f1'])
                    valid_thresholds.append(t)
            
            if f1_scores:
                plt.plot(valid_thresholds, f1_scores, 'o-', label=class_name, linewidth=2)
        
        plt.xlabel('Confidence Threshold')
        plt.ylabel('F1 Score')
        plt.title('F1 Score vs Confidence Threshold by Class')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'f1_by_class.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Confusion matrix at optimal threshold
        best_threshold = max(analysis_results.keys(), 
                           key=lambda t: analysis_results[t]['accuracy'])
        
        confident_results = [r for r in results if r['confidence'] >= best_threshold]
        y_true = [r['true_class'] for r in confident_results]
        y_pred = [r['predicted_class'] for r in confident_results]
        
        cm = confusion_matrix(y_true, y_pred, labels=self.class_names)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title(f'Confusion Matrix (Threshold: {best_threshold})')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Create false positive visualizations
        self.create_false_positive_visualization(results, output_dir, threshold=0.5)
        
        print(f"Visualizations saved to {output_dir}/")
    
    def save_results(self, results, analysis_results, optimal_thresholds, output_dir):
        """Save all results to files"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save detailed results
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(output_dir, 'detailed_results.csv'), index=False)
        
        # Save threshold analysis
        with open(os.path.join(output_dir, 'threshold_analysis.json'), 'w') as f:
            # Convert numpy types to regular Python types for JSON serialization
            serializable_results = {}
            for threshold, metrics in analysis_results.items():
                serializable_results[str(threshold)] = {
                    'accuracy': float(metrics['accuracy']),
                    'total_samples': int(metrics['total_samples']),
                    'coverage': float(metrics['coverage']),
                    'class_metrics': {
                        class_name: {
                            k: float(v) if isinstance(v, (np.float32, np.float64)) else int(v) if isinstance(v, (np.int32, np.int64)) else v
                            for k, v in class_metrics.items()
                        }
                        for class_name, class_metrics in metrics['class_metrics'].items()
                    }
                }
            json.dump(serializable_results, f, indent=2)
        
        # Save optimal thresholds
        with open(os.path.join(output_dir, 'optimal_thresholds.json'), 'w') as f:
            serializable_thresholds = {
                class_name: {
                    'threshold': float(data['threshold']),
                    'f1_score': float(data['f1_score'])
                }
                for class_name, data in optimal_thresholds.items()
            }
            json.dump(serializable_thresholds, f, indent=2)
        
        print(f"Results saved to {output_dir}/")
    
    def run_complete_analysis(self, output_dir, test_size_per_class=100):
        """Run the complete testing and analysis pipeline"""
        print("=== CNN Model Testing and Analysis ===\n")
        
        # Create test set
        print("1. Creating test set...")
        test_data = self.create_test_set(test_size_per_class)
        print(f"Created test set with {len(test_data)} images")
        
        # Make predictions
        print("\n2. Making predictions...")
        results = self.predict_with_confidence(test_data)
        print(f"Completed predictions on {len(results)} images")
        
        # Analyze confidence thresholds
        print("\n3. Analyzing confidence thresholds...")
        analysis_results = self.analyze_confidence_thresholds(results)
        
        # Find optimal thresholds
        print("\n4. Finding optimal thresholds...")
        optimal_thresholds = self.find_optimal_thresholds(analysis_results)
        
        print("\nOptimal confidence thresholds by class:")
        for class_name, data in optimal_thresholds.items():
            print(f"  {class_name}: {data['threshold']:.2f} (F1: {data['f1_score']:.3f})")
        
        # Analyze errors
        print("\n5. Analyzing errors...")
        false_positives, false_negatives = self.analyze_errors(results, threshold=0.5)
        
        print("\nFalse Positive Summary:")
        for class_name, fps in false_positives.items():
            print(f"  {class_name}: {len(fps)} false positives")
        
        print("\nFalse Negative Summary:")
        for class_name, fns in false_negatives.items():
            print(f"  {class_name}: {len(fns)} false negatives")
        
        # Create visualizations
        print("\n6. Creating visualizations...")
        self.create_visualizations(results, analysis_results, output_dir)
        
        # Save results
        print("\n7. Saving results...")
        self.save_results(results, analysis_results, optimal_thresholds, output_dir)
        
        print(f"\n=== Analysis Complete ===")
        print(f"All results saved to: {output_dir}/")
        
        return results, analysis_results, optimal_thresholds

# Example usage
if __name__ == "__main__":
    # Configure your paths here
    MODEL_PATH = "C:/Users/c0062193.CAMPUS/OneDrive - Newcastle University/General - Fox-AI/AI results/Model performance/fox_v20_cleaned_20250603_121146/models/FoxSpot_v20.h5"  # Your trained Keras model
    TRAINING_DIR = "G:/Data/data_v15/Cleaned/"  # Directory with class folders
    REMOVED_DIR = "G:/Data/data_v15/Removed/"    # Directory with removed images (same structure)
    TEST_DIR = "G:/Data/new_fox_test_images/unsplash_fox_2/laura_tests"          # Optional: Directory with dedicated test images (class subfolders)
    OUTPUT_DIR = "G:/Data/new_fox_test_images/test_w_removed"  # Fixed: Use your desired output directory
    
    # Initialize tester
    tester = CNNTester(
        model_path=MODEL_PATH,
        training_dir=TRAINING_DIR,
        removed_dir=REMOVED_DIR,
        test_dir=TEST_DIR,  # Add your test directory here
        img_size=(224, 224)  # Adjust to match your model's input size
    )
    
    # Run complete analysis - FIXED: Use OUTPUT_DIR variable instead of hardcoded string
    results, analysis_results, optimal_thresholds = tester.run_complete_analysis(
        test_size_per_class=100,  # Number of test images per class
        output_dir=OUTPUT_DIR  # Now uses your specified directory
    )
    
    # Optional: Detailed error analysis for specific threshold
    fp_dict, fn_dict = tester.analyze_errors(results, threshold=0.7)
    
    # Print some specific false positive examples for each class
    print("\n=== Detailed Error Analysis ===")
    for class_name in ['fox', 'person', 'badger', 'deer', 'bird', 'squirrel', 'lagomorph']:
        if class_name in fp_dict and fp_dict[class_name]:
            print(f"\nFalse Positives for {class_name}:")
            for fp in fp_dict[class_name][:3]:  # Show first 3 examples
                print(f"  {fp['image_path']} (true: {fp['true_class']}, conf: {fp['confidence']:.3f}, source: {fp['source']})")
        
        if class_name in fn_dict and fn_dict[class_name]:
            print(f"\nFalse Negatives for {class_name}:")
            for fn in fn_dict[class_name][:3]:  # Show first 3 examples
                print(f"  {fn['image_path']} (predicted: {fn['predicted_class']}, conf: {fn['confidence']:.3f}, source: {fn['source']})")