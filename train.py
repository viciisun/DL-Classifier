from tqdm import tqdm
import numpy as np
import os
import json
import time
from datetime import datetime
from model import AdamOptimizer

def train_model(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=128, 
                lr=0.001, momentum=0.9, weight_decay=1e-4, optimizer='sgd',
                early_stopping=True, patience=10,
                save_dir='logs', model_name="default", features=None):
    """
    Train the neural network model
    
    Args:
        model: Neural network model to train
        X_train: Training features
        y_train: Training labels (one-hot encoded)
        X_val: Validation features
        y_val: Validation labels (one-hot encoded)
        epochs: Number of training epochs
        batch_size: Mini-batch size
        lr: Learning rate
        momentum: Momentum coefficient for SGD
        weight_decay: Weight decay (L2 regularization) coefficient
        optimizer: Optimizer type ('sgd' or 'adam')
        early_stopping: Whether to use early stopping
        patience: Number of epochs to wait for improvement before stopping
        save_dir: Directory to save training logs
        model_name: Name of the model
        features: List of features/modules used in the model
        
    Returns:
        Best model parameters
    """
    n = X_train.shape[0]
    best_val_acc = 0.0
    best_model_params = None
    best_epoch = 0
    patience_counter = 0
    train_losses = []
    val_losses = []
    val_accs = []
    
    # Initialize optimizer
    if optimizer == 'adam':
        adam = AdamOptimizer(learning_rate=lr)
    
    # Training start time
    start_time = time.time()
    
    print("\nStarting training:")
    print("-" * 50)
        
    for epoch in range(epochs):
        epoch_start_time = time.time()
        epoch_loss = 0.0
        
        # Shuffle training data
        indices = np.random.permutation(n)
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]
        
        # Mini-batch training
        with tqdm(total=n // batch_size, desc=f"Epoch {epoch+1}/{epochs}") as pbar:
            for i in range(0, n, batch_size):
                X_batch = X_train_shuffled[i:i+batch_size]
                y_batch = y_train_shuffled[i:i+batch_size]
                
                # Forward pass
                probs = model.forward(X_batch, training=True)
                loss = model.compute_loss(probs, y_batch)
                epoch_loss += loss * (X_batch.shape[0] / n)
                
                # Backward pass
                grads = model.backward(X_batch, y_batch)
                
                # Update parameters
                if optimizer == 'sgd':
                    model.update(grads, lr, momentum, weight_decay)
                elif optimizer == 'adam':
                    adam.update(model.layers, grads)
                
                pbar.set_postfix(loss=loss, lr=lr)
                pbar.update(1)
        
        # Track training loss
        train_losses.append(epoch_loss)
                
        # Validation evaluation
        val_probs = model.forward(X_val, training=False)
        val_loss = model.compute_loss(val_probs, y_val)
        val_losses.append(val_loss)
        
        val_preds = np.argmax(val_probs, axis=1)
        val_true = np.argmax(y_val, axis=1)
        val_acc = np.mean(val_preds == val_true)
        val_accs.append(val_acc)
        
        # Calculate epoch runtime
        epoch_time = time.time() - epoch_start_time
        
        print(f"Epoch {epoch+1}: Train Loss={epoch_loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}, LR={lr:.6f}, Time={epoch_time:.2f}s")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_params = [{k: v.copy() for k, v in layer.items()} for layer in model.layers]
            best_epoch = epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1
            
        # Early stopping
        if early_stopping and patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}. Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
            break
    
    # Training end time and total training time
    total_training_time = time.time() - start_time
    
    # Save training history
    history = {
        'model_name': model_name,
        'features': features or [],
        'epochs': epoch + 1,
        'best_epoch': best_epoch,
        'best_val_acc': float(best_val_acc),
        'training_time': float(total_training_time),
        'train_losses': [float(x) for x in train_losses],
        'val_losses': [float(x) for x in val_losses],
        'val_accs': [float(x) for x in val_accs],
        'hyperparameters': {
            'batch_size': batch_size,
            'learning_rate': lr,
            'momentum': momentum,
            'weight_decay': weight_decay,
            'optimizer': optimizer,
            'early_stopping': early_stopping,
            'patience': patience
        },
        'runtime_info': {
            'total_training_time': float(total_training_time),
            'avg_epoch_time': float(total_training_time) / (epoch + 1),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    }
    
    # Save history to JSON file
    history_file = os.path.join(save_dir, f"{model_name}_history.json")
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=4)
    
    print("\nTraining Summary:")
    print("-" * 50)
    print(f"Total training time: {total_training_time:.2f} seconds")
    print(f"Average time per epoch: {total_training_time / (epoch + 1):.2f} seconds")
    print(f"Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
    print(f"Training history saved to {history_file}")
    
    return best_model_params, save_dir