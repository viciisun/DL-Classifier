import matplotlib.pyplot as plt
import numpy as np
import os
import json
from datetime import datetime

def plot_confusion_matrix(cm, class_names=None, save_path=None):
    """
    Plot confusion matrix
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        save_path: Path to save the figure
    """
    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]
        
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, cmap='Blues')
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylim(len(cm) - 0.5, -0.5)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()
    
    plt.close()

def plot_training_history(history_file, save_dir='logs'):
    """
    Plot training history
    
    Args:
        history_file: Path to history JSON file
        save_dir: Directory to save the figures
    """
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    model_name = history['model_name']
    epochs = range(1, len(history['train_losses']) + 1)
    
    # Plot training and validation loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_losses'], 'b', label='Training loss')
    plt.plot(epochs, history['val_losses'], 'r', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['val_accs'], 'g', label='Validation accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    
    # Save figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f"{model_name}_{timestamp}_history.png")
    plt.savefig(save_path)
    print(f"Training history plot saved to {save_path}")
    plt.close()
    
    return save_path

def plot_model_comparison(log_dir='logs', metric='accuracy', top_n=None, save_dir='logs'):
    """
    Plot model comparison
    
    Args:
        log_dir: Directory containing evaluation logs
        metric: Metric to compare ('accuracy', 'f1_score', 'precision', 'recall')
        top_n: Number of top models to display
        save_dir: Directory to save the figure
    """
    metrics_files = [f for f in os.listdir(log_dir) if f.endswith('_metrics.json')]
    
    if not metrics_files:
        print("No metrics files found")
        return
    
    all_metrics = []
    for file in metrics_files:
        with open(os.path.join(log_dir, file), 'r') as f:
            metrics = json.load(f)
            all_metrics.append(metrics)
    
    # Sort by the specified metric
    all_metrics.sort(key=lambda x: x[metric], reverse=True)
    
    if top_n:
        all_metrics = all_metrics[:top_n]
    
    # Plot bar chart
    plt.figure(figsize=(12, 6))
    models = [m['model_name'] for m in all_metrics]
    values = [m[metric] for m in all_metrics]
    
    bars = plt.bar(models, values, color='skyblue')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom')
    
    plt.title(f'Model Comparison - {metric.capitalize()}')
    plt.xlabel('Model')
    plt.ylabel(metric.capitalize())
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f"model_comparison_{metric}_{timestamp}.png")
    plt.savefig(save_path)
    print(f"Model comparison plot saved to {save_path}")
    plt.close()
    
    return save_path

def plot_feature_importance(pca_explained_variance, save_dir='logs'):
    """
    Plot PCA feature importance
    
    Args:
        pca_explained_variance: PCA explained variance ratio
        save_dir: Directory to save the figure
    """
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(pca_explained_variance)), pca_explained_variance, alpha=0.5, align='center')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.title('Explained Variance Ratio of Principal Components')
    
    # Save figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f"pca_explained_variance_{timestamp}.png")
    plt.savefig(save_path)
    print(f"PCA explained variance plot saved to {save_path}")
    plt.close()
    
    return save_path

def visualize_all_results(log_dir='logs'):
    """
    Visualize all results
    
    Args:
        log_dir: Directory containing evaluation logs
    """
    # Plot training history for all history files
    history_files = [f for f in os.listdir(log_dir) if f.endswith('_history.json')]
    for file in history_files:
        plot_training_history(os.path.join(log_dir, file), log_dir)
    
    # Plot model comparison for different metrics
    metrics = ['accuracy', 'f1_score', 'precision', 'recall']
    for metric in metrics:
        plot_model_comparison(log_dir, metric, save_dir=log_dir)
    
    print("All visualizations generated successfully")