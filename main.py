import argparse
import sys
import os
import numpy as np
from data_loader import load_and_preprocess_data
from model import NeuralNetwork
from train import train_model
from evaluate import evaluate_model, ModelEvaluator
from visualize import plot_confusion_matrix
from preprocess import DataPreprocessor
import time
import platform
import json
from datetime import datetime

def get_system_info():
    """Get system information for the report"""
    return {
        "os": platform.system(),
        "os_version": platform.version(),
        "python_version": platform.python_version(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "memory": platform.machine(),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Neural Network Classifier')
    
    # Model configuration
    parser.add_argument('features', nargs='*', default=['default'],
                        help='List of features to use (preprocess, GELU, Adam, or full)')
    
    # Hyperparameters
    parser.add_argument('--hidden_sizes', type=int, nargs='+', default=[256, 128],
                        help='List of hidden layer sizes')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Mini-batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum coefficient for SGD')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay (L2 regularization) coefficient')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout probability')
    
    # Preprocessing
    parser.add_argument('--preprocess_method', type=str, default='standard',
                        choices=['standard', 'minmax', 'pca'],
                        help='Preprocessing method')
    parser.add_argument('--pca_components', type=int, default=None,
                        help='Number of PCA components')
    
    # Output
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory to save logs')
    
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # Create base log directory
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Save system information
    sys_info = get_system_info()
    with open(os.path.join(args.log_dir, 'system_info.json'), 'w') as f:
        json.dump(sys_info, f, indent=4)
    
    # Determine which features to use
    use_preprocess = 'preprocess' in args.features or 'full' in args.features
    use_gelu = 'GELU' in args.features or 'full' in args.features
    use_adam = 'Adam' in args.features or 'full' in args.features
    
    # Create model name based on features
    if 'full' in args.features:
        model_name = "full_model"
        features = ["preprocess", "GELU", "Adam"]
    elif 'default' in args.features and len(args.features) == 1:
        model_name = "default_model"
        features = []
    else:
        model_name = "_".join(args.features)
        features = args.features
    
    # Print configuration
    print(f"Running model: {model_name}")
    print(f"Features: {features}")
    print(f"Hidden sizes: {args.hidden_sizes}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    
    # Load data
    print("Loading data...")
    X_train_full, y_train_onehot, X_test, y_test_onehot, y_test = load_and_preprocess_data()
    
    # Split training data into train and validation sets (80% train, 20% validation)
    n_train = int(0.8 * X_train_full.shape[0])
    X_train = X_train_full[:n_train]
    y_train = y_train_onehot[:n_train]
    X_val = X_train_full[n_train:]
    y_val = y_train_onehot[n_train:]
    
    # Preprocess data if requested
    if use_preprocess:
        print(f"Preprocessing data using {args.preprocess_method}...")
        preprocessor = DataPreprocessor(method=args.preprocess_method, n_components=args.pca_components)
        X_train = preprocessor.fit_transform(X_train)
        X_val = preprocessor.transform(X_val)
        X_test = preprocessor.transform(X_test)
        
        # Save preprocessor details
        if args.preprocess_method == 'pca' and args.pca_components:
            print(f"PCA explained variance ratio: {preprocessor.get_explained_variance_ratio()[:10]}")
    
    # Initialize model
    activation = 'gelu' if use_gelu else 'relu'
    print(f"Initializing model with activation: {activation}")
    model = NeuralNetwork(
        input_size=X_train.shape[1],
        hidden_sizes=args.hidden_sizes,
        output_size=10,
        use_bn=True,
        dropout_prob=args.dropout,
        activation=activation
    )
    
    # Training parameters
    optimizer = 'adam' if use_adam else 'sgd'
    
    print(f"Training with optimizer: {optimizer}")
    
    # Start training time
    start_time = time.time()
    
    # Train model
    best_model_params, log_subdir = train_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        optimizer=optimizer,
        lr_scheduler=None,  # No learning rate scheduling for simplicity
        early_stopping=True,  # Always use early stopping
        patience=10,
        save_dir=args.log_dir,
        model_name=model_name,
        features=features
    )
    
    # End training time
    training_time = time.time() - start_time
    
    # Load best model
    model.layers = best_model_params
    
    # Evaluate test set
    print("Evaluating model on test set...")
    evaluator = ModelEvaluator(log_dir=log_subdir)
    metrics = evaluator.evaluate(
        model=model,
        X_test=X_test,
        y_test=y_test_onehot,
        model_name=model_name,
        features=features,
        verbose=True
    )
    
    # Save training summary
    summary = {
        'model_name': model_name,
        'features': features,
        'test_accuracy': float(metrics['accuracy']),
        'test_precision': float(metrics['precision']),
        'test_recall': float(metrics['recall']),
        'test_f1_score': float(metrics['f1_score']),
        'training_time': float(training_time),
        'inference_time': float(metrics['inference_time']),
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'configuration': vars(args),
        'system_info': sys_info
    }
    
    summary_file = os.path.join(log_subdir, f"{model_name}_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=4)
    
    print(f"Summary saved to {summary_file}")
    print(f"Total training time: {training_time:.2f} seconds")
    print(f"Test accuracy: {metrics['accuracy']:.4f}")
    
    # Compare with other models if available
    try:
        all_models = evaluator.compare_models(log_dir=args.log_dir)
        print(f"\nNumber of models compared: {len(all_models)}")
    except Exception as e:
        print(f"Error comparing models: {e}")
        print("This could happen if this is the first model you're running.")