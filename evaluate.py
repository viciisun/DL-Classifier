import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    confusion_matrix,
    classification_report
)
import time
import os
import json
from datetime import datetime

class ModelEvaluator:
    def __init__(self, log_dir='logs'):
        """
        Initialize the model evaluator
        
        Args:
            log_dir (str): Directory to store evaluation logs
        """
        self.log_dir = log_dir
        
        # Create base log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Create date-based directory for regular training
        if 'ablation' not in log_dir:
            date = datetime.now().strftime("%Y%m%d")
            
            # Find the next sequence number
            try:
                existing_runs = [d for d in os.listdir(log_dir) if d.startswith(date) and '_' in d]
                if existing_runs:
                    last_seq = max([int(d.split('_')[1]) for d in existing_runs if d.split('_')[1].isdigit()])
                    seq_num = f"{last_seq + 1:03d}"
                else:
                    seq_num = "001"
            except Exception as e:
                print(f"Warning: Error reading directory {log_dir}: {str(e)}")
                seq_num = "001"
                
            self.log_subdir = os.path.join(log_dir, f"{date}_{seq_num}")
        else:
            # For ablation studies, use the provided directory
            self.log_subdir = log_dir
            
        # Create the final log subdirectory
        os.makedirs(self.log_subdir, exist_ok=True)
        print(f"Logs will be saved to: {self.log_subdir}")
        
    def evaluate(self, model, X_test, y_test, model_name="default", features=None, verbose=True):
        """
        Evaluate the model and calculate various metrics
        
        Args:
            model: Trained neural network model
            X_test: Test features
            y_test: Test labels (one-hot encoded)
            model_name (str): Name of the model
            features (list): List of features/modules used in the model
            verbose (bool): Whether to print evaluation results
            
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        # Start timing for inference
        start_time = time.time()
        
        # Get predictions
        probs = model.forward(X_test, training=False)
        y_pred = np.argmax(probs, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        # Calculate metrics
        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        loss = model.compute_loss(probs, y_test)
        cm = confusion_matrix(y_true, y_pred)
        
        # End timing for inference
        inference_time = time.time() - start_time
        
        # Create metrics dictionary
        metrics = {
            'model_name': model_name,
            'features': features or [],
            'accuracy': float(acc),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'loss': float(loss),
            'inference_time': float(inference_time),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        
        # Print evaluation results
        if verbose:
            print(f"Model: {model_name}")
            print(f"Features: {features or []}")
            print(f"Accuracy: {acc:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")
            print(f"Loss: {loss:.4f}")
            print(f"Inference Time: {inference_time:.4f} seconds")
            print("\nClassification Report:")
            print(classification_report(y_true, y_pred, zero_division=0))
        
        # Save metrics to log file
        self._save_metrics(metrics, cm)
        
        return metrics
    
    def visualize_confusion_matrix(self, cm, class_names=None):
        """
        Visualize confusion matrix
        
        Args:
            cm: Confusion matrix
            class_names (list): List of class names
        """
        if class_names is None:
            class_names = [str(i) for i in range(cm.shape[0])]
        
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        
        # Save figure
        save_path = os.path.join(self.log_subdir, f"confusion_matrix.png")
        plt.savefig(save_path)
        plt.close()
        
        return save_path
    
    def _save_metrics(self, metrics, confusion_matrix):
        """
        Save metrics to log file
        
        Args:
            metrics: Dictionary of evaluation metrics
            confusion_matrix: Confusion matrix
        """
        model_name = metrics['model_name']
        
        # Save metrics to JSON file
        metrics_file = os.path.join(self.log_subdir, f"{model_name}_metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # Save confusion matrix
        np.save(os.path.join(self.log_subdir, f"{model_name}_confusion_matrix.npy"), confusion_matrix)
        
        # Visualize confusion matrix
        cm_path = self.visualize_confusion_matrix(confusion_matrix)
        
        print(f"Evaluation metrics saved to {metrics_file}")
        print(f"Confusion matrix visualization saved to {cm_path}")
    
    @staticmethod
    def compare_models(log_dir='logs'):
        """
        Compare different models based on saved metrics
        
        Args:
            log_dir (str): Directory containing evaluation logs
            
        Returns:
            list: List of model metrics for comparison
        """
        # Find all metrics files in all subdirectories
        all_metrics = []
        for root, dirs, files in os.walk(log_dir):
            for file in files:
                if file.endswith('_metrics.json'):
                    try:
                        metrics_path = os.path.join(root, file)
                        with open(metrics_path, 'r') as f:
                            metrics = json.load(f)
                            
                        # Try to get hyperparameters from summary file
                        summary_path = os.path.join(root, file.replace('_metrics.json', '_summary.json'))
                        if os.path.exists(summary_path):
                            with open(summary_path, 'r') as f:
                                summary = json.load(f)
                                if 'configuration' in summary:
                                    metrics['hyperparameters'] = summary['configuration']
                                if 'preprocess_method' in summary:
                                    metrics['preprocess_method'] = summary['preprocess_method']
                                    
                        # Get directory name (date_sequence)
                        dir_name = os.path.basename(os.path.dirname(metrics_path))
                        metrics['dir_name'] = dir_name
                                    
                        all_metrics.append(metrics)
                    except Exception as e:
                        print(f"Error loading metrics from {metrics_path}: {str(e)}")
        
        if not all_metrics:
            print("No metrics files found")
            return []
        
        # Sort by accuracy
        all_metrics.sort(key=lambda x: x.get('accuracy', 0), reverse=True)
        
        # Print comparison
        print("\nModel Comparison:")
        headers = ['Run', 'Accuracy', 'F1 Score', 'Preprocess', 'Activation', 'Optimizer', 'Dropout', 'LR', 'Batch', 'Hidden Sizes']
        row_format = "{:<12} {:<10} {:<10} {:<11} {:<11} {:<10} {:<10} {:<10} {:<8} {:<20}"
        
        print(row_format.format(*headers))
        print("-" * 120)
        
        for m in all_metrics:
            # Get hyperparameters
            hp = m.get('hyperparameters', {})
            features = set(m.get('features', []))
            
            # Get preprocessing method
            preprocess = m.get('preprocess_method', 'standard').title()
            
            # Determine settings
            is_gelu = 'GELU' in features
            is_adam = 'Adam' in features
            
            hidden_sizes = str(hp.get('hidden_sizes', 'N/A'))
            if len(hidden_sizes) > 17:
                hidden_sizes = hidden_sizes[:14] + '...'
                
            print(row_format.format(
                m.get('dir_name', 'Unknown'),
                f"{m.get('accuracy', 0):.4f}",
                f"{m.get('f1_score', 0):.4f}",
                preprocess,
                'GELU' if is_gelu else 'ReLU',
                'Adam' if is_adam else 'SGD',
                str(hp.get('dropout', 'N/A')),
                str(hp.get('lr', 'N/A')),
                str(hp.get('batch_size', 'N/A')),
                hidden_sizes
            ))
        
        print("\nDetailed metrics saved in respective log directories")
        return all_metrics

# For backward compatibility
def evaluate_model(model, X_test, y_test):
    """
    Legacy function for model evaluation
    
    Args:
        model: Trained neural network model
        X_test: Test features
        y_test: Test labels
        
    Returns:
        tuple: (accuracy, confusion_matrix)
    """
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate(model, X_test, y_test, verbose=False)
    probs = model.forward(X_test, training=False)
    y_pred = np.argmax(probs, axis=1)
    y_true = np.argmax(y_test, axis=1)
    cm = confusion_matrix(y_true, y_pred)
    return metrics['accuracy'], cm