import sys
import os
import numpy as np
import platform
import time
from data_loader import load_and_preprocess_data
from model import NeuralNetwork
from train import train_model
from evaluate import evaluate_model, ModelEvaluator
from preprocess import DataPreprocessor
import json
from datetime import datetime
import psutil
import subprocess
from cli import parse_arguments, generate_model_name

def get_system_info():
    """Get system information for the report"""
    # Get basic system info
    system = platform.system()
    version = platform.version()
    python_version = platform.python_version()
    
    # Get processor info for Apple Silicon
    if system == "Darwin":
        try:
            # Get processor info using sysctl
            processor_info = subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string']).decode().strip()
            if "Apple" in processor_info:
                processor = processor_info
            else:
                processor = platform.processor()
        except:
            processor = platform.processor()
    else:
        processor = platform.processor()
    
    # Get memory info
    memory = psutil.virtual_memory()
    memory_info = f"{memory.total / (1024**3):.1f}GB"
    
    return {
        "os": system,
        "os_version": version,
        "python_version": python_version,
        "processor": processor,
        "cpu_count": psutil.cpu_count(),
        "memory": memory_info,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Initialize evaluator first to get the log directory
    evaluator = ModelEvaluator(log_dir=args.log_dir)
    log_dir = evaluator.log_subdir
    
    # Generate model name
    model_name = generate_model_name(args)

    # Print model configuration
    print("\n" + "="*50)
    print(f"Run Configuration ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
    print("="*50)
    
    print("\nPreprocessing:")
    print(f"  Method: {args.preprocess_method}")
    if args.preprocess_method == 'standard':
        print("  Parameters: mean=0, std=1")
    elif args.preprocess_method == 'minmax':
        print("  Parameters: range=[0,1]")
    
    print("\nModel Architecture:")
    print(f"  Input Size: {128}")  # Default input size
    print(f"  Hidden Sizes: {args.hidden_sizes}")
    print(f"  Output Size: {10}")  # Default output size for MNIST
    print(f"  Activation: {'GELU' if args.gelu else 'ReLU'}")
    print(f"  Dropout Rate: {args.dropout}")
    
    print("\nTraining Parameters:")
    print(f"  Optimizer: {'Adam' if args.adam else 'SGD'}")
    print(f"  Learning Rate: {args.lr}")
    if not args.adam:
        print(f"  Momentum: {args.momentum}")
    print(f"  Weight Decay: {args.weight_decay}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Max Epochs: {args.epochs}")
    print(f"  Early Stopping Patience: {args.early_stopping_patience}")
    
    print("\nSystem Information:")
    print(f"  Python Version: {sys.version.split()[0]}")
    print(f"  NumPy Version: {np.__version__}")
    print(f"  Device: CPU")
    print("="*50 + "\n")
    
    # Save system information
    sys_info = get_system_info()
    with open(os.path.join(log_dir, 'system_info.json'), 'w') as f:
        json.dump(sys_info, f, indent=4)
    
    # Save configuration for model comparison
    config = {
        'model_name': model_name,
        'preprocess_method': args.preprocess_method,
        'features': args.features,
        'configuration': vars(args),
        'system_info': sys_info,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(os.path.join(log_dir, f'{model_name}_summary.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    # Load data
    print("\nLoading data...")
    X_train_full, y_train_onehot, X_test, y_test_onehot, y_test = load_and_preprocess_data()
    
    # Split training data into train and validation sets (80% train, 20% validation)
    n_train = int(0.8 * X_train_full.shape[0])
    X_train = X_train_full[:n_train]
    y_train = y_train_onehot[:n_train]
    X_val = X_train_full[n_train:]
    y_val = y_train_onehot[n_train:]
    
    # Preprocess data
    print(f"Preprocessing data using {args.preprocess_method} scaling...")
    preprocessor = DataPreprocessor(method=args.preprocess_method)
    X_train = preprocessor.fit_transform(X_train)
    X_val = preprocessor.transform(X_val)
    X_test = preprocessor.transform(X_test)
    
    # Save preprocessor parameters
    preprocess_info = preprocessor.get_params()
    with open(os.path.join(log_dir, 'preprocess_info.json'), 'w') as f:
        json.dump(preprocess_info, f, indent=4)
    
    # Initialize model
    activation = 'gelu' if args.gelu else 'relu'
    model = NeuralNetwork(
        input_size=X_train.shape[1],
        hidden_sizes=args.hidden_sizes,
        output_size=10,
        use_bn=args.use_bn,
        dropout_prob=args.dropout,
        activation=activation
    )
    
    # Training parameters
    optimizer = 'adam' if args.adam else 'sgd'
    
    # Start training time
    start_time = time.time()
    
    # Train model
    best_model_params, _ = train_model(
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
        early_stopping=True,
        patience=args.early_stopping_patience,
        save_dir=log_dir,
        model_name=model_name,
        features=args.features
    )
    
    # End training time
    training_time = time.time() - start_time
    
    # Load best model
    model.layers = best_model_params
    
    # Evaluate test set
    print("\nEvaluating model on test set...")
    metrics = evaluator.evaluate(
        model=model,
        X_test=X_test,
        y_test=y_test_onehot,
        model_name=model_name,
        features=args.features,
        verbose=True
    )
    
    print(f"\nTotal training time: {training_time:.2f} seconds")
    print(f"Test accuracy: {metrics['accuracy']:.4f}")
    
    # Compare with other models if available
    try:
        all_models = evaluator.compare_models(args.log_dir)
        print(f"\nNumber of models compared: {len(all_models)}")
    except Exception as e:
        print(f"Error comparing models: {e}")
        print("This could happen if this is the first model you're running.")

if __name__ == "__main__":
    main()