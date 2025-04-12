import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
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
        # Create timestamp directory for this evaluation
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_subdir = os.path.join(log_dir, self.timestamp)
        os.makedirs(self.log_subdir, exist_ok=True)
        
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
            'timestamp': self.timestamp,
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
                        with open(os.path.join(root, file), 'r') as f:
                            metrics = json.load(f)
                            all_metrics.append(metrics)
                    except:
                        print(f"Error loading metrics from {os.path.join(root, file)}")
        
        if not all_metrics:
            print("No metrics files found")
            return []
        
        # Sort by accuracy
        all_metrics.sort(key=lambda x: x.get('accuracy', 0), reverse=True)
        
        # Print comparison
        print("Model Comparison:")
        print(f"{'Model':<20} {'Features':<40} {'Accuracy':<10} {'Inference Time':<15}")
        print('-' * 85)
        
        for m in all_metrics:
            features_str = ', '.join(m.get('features', [])) if m.get('features', []) else 'None'
            if len(features_str) > 37:
                features_str = features_str[:34] + '...'
            print(f"{m.get('model_name', 'Unknown'):<20} {features_str:<40} {m.get('accuracy', 0):<10.4f} {m.get('inference_time', 0):<15.4f}s")
        
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