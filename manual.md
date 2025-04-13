# Deep Learning Classifier Manual

## Table of Contents

1. [Introduction](#introduction)
2. [Installation and Setup](#installation-and-setup)
3. [Codebase Structure](#codebase-structure)
4. [Core Components in Detail](#core-components-in-detail)
   - [Neural Network Model](#neural-network-model)
   - [Activation Functions](#activation-functions)
   - [Optimizers](#optimizers)
   - [Regularization Techniques](#regularization-techniques)
   - [Data Preprocessing](#data-preprocessing)
5. [Command Line Arguments in Detail](#command-line-arguments-in-detail)
6. [Training Process](#training-process)
7. [Evaluation and Visualization](#evaluation-and-visualization)
8. [Ablation Studies](#ablation-studies)
9. [Hyperparameter Analysis](#hyperparameter-analysis)
10. [Performance Optimization Tips](#performance-optimization-tips)
11. [Frequently Asked Questions](#frequently-asked-questions)
12. [Appendix: Complete API Reference](#appendix-complete-api-reference)

## Introduction

Deep Learning Classifier is a neural network classifier framework implemented from scratch using NumPy, without relying on any deep learning frameworks. This project is an assignment for the COMP4329 Deep Learning course, designed to help users deeply understand the internal workings of neural networks.

This manual will provide detailed instructions on how to use this framework to train and evaluate neural network models, and how to adjust various parameters to achieve optimal performance.

## Installation and Setup

### System Requirements

- Python 3.6+
- Recommended: 4+ core CPU, 8GB+ RAM

### Dependencies

This project depends on the following Python libraries:

```
numpy
matplotlib
scikit-learn
tqdm
psutil
```

### Installation Steps

1. Clone or download this repository
2. Install required dependencies:

```bash
pip install numpy matplotlib scikit-learn tqdm psutil
```

3. Prepare the dataset (MNIST dataset by default, needs to be saved in NumPy format):
   - `data/train_data.npy`: Training data features
   - `data/train_label.npy`: Training data labels
   - `data/test_data.npy`: Test data features
   - `data/test_label.npy`: Test data labels

## Codebase Structure

This codebase consists of the following main files:

- `main.py`: Main script for running the model
- `cli.py`: Command line interface and parameter parsing
- `model.py`: Neural network implementation
- `train.py`: Training functionality
- `evaluate.py`: Evaluation metrics and visualization
- `data_loader.py`: Data loading and initial preprocessing
- `preprocess.py`: Data preprocessing (standardization and min-max scaling)
- `visualize.py`: Visualization tools

## Core Components in Detail

### Neural Network Model

The core implementation is in the `NeuralNetwork` class in the `model.py` file.

#### Architecture Design

The neural network consists of the following components:

- **Input Layer**: Receives input features
- **Hidden Layers**: Customizable number and size of hidden layers
- **Output Layer**: Produces class probabilities

Each hidden layer performs the following operations in sequence:

1. Linear transformation (weights and biases)
2. Batch normalization (if enabled)
3. Activation function
4. Dropout (during training)

#### Initialization Method

```python
def __init__(self, input_size=128, hidden_sizes=[256, 128], output_size=10,
             use_bn=True, dropout_prob=0.5, activation='relu')
```

Parameter descriptions:

- `input_size`: Dimension of input features
- `hidden_sizes`: List of hidden layer sizes, e.g., [256, 128] indicates two hidden layers
- `output_size`: Number of output classes
- `use_bn`: Whether to use batch normalization
- `dropout_prob`: Dropout probability (0-1)
- `activation`: Activation function ('relu' or 'gelu')

#### Weight Initialization

The model uses Xavier initialization (also known as Glorot initialization) to set initial weights:

```python
w = np.random.randn(sizes[i], sizes[i+1]) * np.sqrt(2.0 / sizes[i])
```

This initialization method helps prevent gradient vanishing or exploding problems in deep networks.

### Activation Functions

Two activation functions are implemented:

#### ReLU (Rectified Linear Unit)

```python
def relu(self, x):
    """ReLU activation function"""
    return np.maximum(0, x)
```

Advantages of ReLU:

- Simple and computationally efficient
- Effectively mitigates the vanishing gradient problem
- Leads to sparse network activation, improving computational efficiency

Disadvantages:

- "Dying ReLU" problem: When input is negative, the gradient is zero, and neurons may permanently stop learning

#### GELU (Gaussian Error Linear Unit)

```python
def gelu(x):
    """GELU activation function: x * Φ(x)"""
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))
```

Advantages of GELU:

- Provides smooth non-linearity near zero
- Reduces "dead neuron" problem
- Performs well in modern Transformer architectures

### Optimizers

Two optimization algorithms are implemented:

#### SGD with Momentum

```python
def update(self, grads, lr, momentum, weight_decay):
    """Update parameters using SGD with momentum"""
    # Implementation code...
```

Parameters:

- `lr`: Learning rate
- `momentum`: Momentum coefficient
- `weight_decay`: Weight decay coefficient (L2 regularization)

How it works:

1. Calculate current gradients (considering weight decay)
2. Update velocity vector with momentum
3. Update weights using velocity vector

#### Adam Optimizer

```python
class AdamOptimizer:
    def __init__(self, learning_rate=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8):
        # Initialization code...

    def update(self, layers, grads):
        # Implementation code...
```

Parameters:

- `learning_rate`: Learning rate
- `beta1`: Exponential decay rate for first moment estimates
- `beta2`: Exponential decay rate for second moment estimates
- `epsilon`: Small constant for numerical stability

How it works:

1. Calculate first moment (mean) and second moment (uncentered variance) of gradients
2. Apply bias correction
3. Update each parameter with adaptive learning rate

### Regularization Techniques

Three regularization techniques are implemented:

#### Dropout

```python
# Implementation in forward pass
if training and self.dropout_prob > 0:
    mask = (np.random.rand(*h.shape) > self.dropout_prob) / (1 - self.dropout_prob)
    h *= mask
    self.cache[-1]['mask'] = mask
```

Parameters:

- `dropout_prob`: Probability of dropping neurons

How it works:

- Randomly deactivates a fraction of neurons during training
- Scales by dividing by (1-dropout_prob) to ensure the expected value remains unchanged
- Keeps all neurons active during testing

#### Batch Normalization

```python
def batch_norm_forward(self, x, gamma, beta, running_mean, running_var, training, momentum=0.9):
    """Batch normalization forward pass"""
    # Implementation code...
```

Parameters:

- `gamma`: Scale parameter
- `beta`: Shift parameter
- `running_mean`: Running mean (for inference)
- `running_var`: Running variance (for inference)
- `momentum`: Momentum for running statistics

How it works:

- During training: Compute mean and variance of mini-batch, normalize input, apply scale and shift
- During inference: Use running statistics for normalization
- Reduces internal covariate shift, speeds up training

#### Weight Decay (L2 Regularization)

```python
# Implementation in update
grads[i]['W'] += weight_decay * layer['W']
```

Parameters:

- `weight_decay`: Weight decay coefficient

How it works:

- Adds a penalty term to the loss function proportional to the square of weights
- Encourages the model to learn smaller weights
- Reduces model complexity, prevents overfitting

### Data Preprocessing

Implemented in `preprocess.py`, the `DataPreprocessor` class provides different preprocessing methods.

#### Standard Scaling

```python
def fit(self, X):
    """Fit the preprocessor on training data"""
    if self.method == 'standard':
        self.scaler = StandardScaler()
        self.scaler.fit(X)
    # Other methods...
```

How it works:

- Adjusts each feature to have zero mean and unit variance
- Formula: x_normalized = (x - mean) / std
- Makes all features on the same scale

#### Min-Max Scaling

```python
def fit(self, X):
    """Fit the preprocessor on training data"""
    if self.method == 'minmax':
        self.scaler = MinMaxScaler()
        self.scaler.fit(X)
    # Other methods...
```

How it works:

- Scales features to [0,1] range
- Formula: x_normalized = (x - min) / (max - min)
- Preserves the characteristics of the original distribution

## Command Line Arguments in Detail

Implemented in `cli.py`, providing a user-friendly command line interface to configure the model.

### Basic Usage

```bash
python3 main.py --preprocess standard --gelu --adam
```

### All Available Parameters

| Parameter                   | Type     | Default                  | Description                                           |
| --------------------------- | -------- | ------------------------ | ----------------------------------------------------- |
| `--preprocess`              | string   | 'none'                   | Preprocessing method: 'standard', 'minmax', or 'none' |
| `--gelu`                    | flag     | False                    | Use GELU activation instead of ReLU                   |
| `--adam`                    | flag     | False                    | Use Adam optimizer instead of SGD                     |
| `--no_batch_norm`           | flag     | False                    | Disable batch normalization                           |
| `--hidden_sizes`            | int list | [256, 128]               | List of hidden layer sizes                            |
| `--epochs`                  | int      | 100                      | Number of training epochs                             |
| `--batch_size`              | int      | 128                      | Mini-batch size                                       |
| `--lr`                      | float    | 1e-3 (Adam) / 1e-2 (SGD) | Learning rate                                         |
| `--momentum`                | float    | 0.9                      | Momentum coefficient for SGD                          |
| `--weight_decay`            | float    | 1e-4                     | Weight decay coefficient                              |
| `--dropout`                 | float    | 0.5                      | Dropout probability                                   |
| `--early_stopping_patience` | int      | 10                       | Early stopping patience value                         |
| `--log_dir`                 | string   | 'logs'                   | Directory to save logs                                |
| `--model_name`              | string   | None                     | Custom model name                                     |

### Example Commands

Using Adam optimizer and GELU activation, batch size 64, learning rate 0.0005:

```bash
python3 main.py --preprocess standard --gelu --adam --batch_size 64 --lr 0.0005
```

Using a deeper network architecture:

```bash
python3 main.py --hidden_sizes 512 256 128 64
```

## Training Process

The training process is implemented in the `train_model` function in `train.py`.

### Training Steps

1. **Initialization**: Set up best validation accuracy tracker and patience counter
2. **Training Loop**: Iterate for the specified number of epochs
   - Shuffle training data
   - Process data in batches
   - Compute predictions and loss with forward pass
   - Compute gradients with backward pass
   - Update parameters using the optimizer
   - Evaluate performance on validation set
   - Check early stopping condition
3. **Save Results**: Save training history and best model parameters

### Early Stopping Mechanism

```python
if early_stopping and patience_counter >= patience:
    print(f"\nEarly stopping at epoch {epoch+1}")
    break
```

How it works:

- Track best validation accuracy
- Increase patience counter when validation performance doesn't improve
- Stop training when patience is exhausted

### Training Logs

The training process generates detailed logs:

```python
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
```

## Evaluation and Visualization

Implemented in `evaluate.py` and `visualize.py`.

### Evaluation Metrics

```python
# Calculate metrics
acc = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
```

Including:

- **Accuracy**: Proportion of correctly classified samples
- **Precision**: Proportion of true positive predictions among all positive predictions
- **Recall**: Proportion of true positives correctly identified
- **F1 Score**: Harmonic mean of precision and recall

### Confusion Matrix

```python
cm = confusion_matrix(y_true, y_pred)
```

Visualizing the confusion matrix:

- Shows model performance across different classes
- Helps identify classes that are frequently confused

### Training History Visualization

```python
def plot_training_history(history_file, save_dir='logs'):
    # Implementation code...
```

Generates:

- Training and validation loss curves
- Validation accuracy curve

### Model Comparison

```python
def plot_model_comparison(log_dir='logs', metric='accuracy', top_n=None, save_dir='logs'):
    # Implementation code...
```

Displays:

- Comparison of different model configurations on selected metrics
- Helps identify the best model settings

## Ablation Studies

Ablation studies analyze the contribution of each component by systematically removing or replacing model components.

### Components to Analyze

1. **Activation Functions**: ReLU vs. GELU
2. **Optimizers**: SGD+Momentum vs. Adam
3. **Batch Normalization**: Enabled vs. Disabled
4. **Dropout**: Different Dropout probabilities
5. **Network Architecture**: Different number of layers and sizes
6. **Preprocessing Methods**: Standardization vs. Min-Max Scaling

### How to Run Ablation Studies

Run different configurations separately:

```bash
# Baseline model
python3 main.py --preprocess standard --gelu --adam

# Without batch normalization
python3 main.py --preprocess standard --gelu --adam --no_batch_norm

# Using ReLU instead of GELU
python3 main.py --preprocess standard --adam

# Using SGD instead of Adam
python3 main.py --preprocess standard --gelu

# Different Dropout probabilities
python3 main.py --preprocess standard --gelu --adam --dropout 0.3
python3 main.py --preprocess standard --gelu --adam --dropout 0.0
```

## Hyperparameter Analysis

Hyperparameter analysis explores the effect of different hyperparameter settings on model performance.

### Key Hyperparameters

1. **Learning Rate**: Affects training convergence speed and stability
2. **Batch Size**: Affects optimization process and generalization performance
3. **Hidden Layer Size**: Affects model capacity and computational requirements
4. **Dropout Probability**: Affects regularization strength
5. **Weight Decay Coefficient**: Controls L2 regularization strength

### Hyperparameter Search Examples

Learning rate analysis:

```bash
python3 main.py --preprocess standard --gelu --adam --lr 0.0001
python3 main.py --preprocess standard --gelu --adam --lr 0.001
python3 main.py --preprocess standard --gelu --adam --lr 0.01
```

Batch size analysis:

```bash
python3 main.py --preprocess standard --gelu --adam --batch_size 32
python3 main.py --preprocess standard --gelu --adam --batch_size 64
python3 main.py --preprocess standard --gelu --adam --batch_size 128
python3 main.py --preprocess standard --gelu --adam --batch_size 256
```

## Performance Optimization Tips

### Computational Optimization

1. **Use Vectorized Operations**: Avoid Python loops, use NumPy matrix operations when possible
2. **Reduce Memory Copies**: Perform in-place operations when possible
3. **Batch Processing**: Use appropriate batch sizes to balance memory usage and computational efficiency

### Training Optimization

1. **Learning Rate Scheduling**: Gradually reducing learning rate can improve final performance
2. **Gradient Clipping**: Prevents gradient explosion problems
3. **Proper Weight Initialization**: Use Xavier/Glorot initialization to avoid gradient problems

### Memory Optimization

1. **Reduce Intermediate Result Storage**: Only save intermediate values needed for backpropagation
2. **Batch Data Loading**: Avoid loading the entire dataset into memory at once

## Frequently Asked Questions

### 1. Model training is too slow, how can I speed it up?

- Try reducing network size (fewer layers or fewer neurons per layer)
- Increase batch size (if memory allows)
- Reduce preprocessing complexity
- Use simpler activation functions (ReLU is computationally more efficient than GELU)

### 2. Model is overfitting, how can I fix it?

- Increase Dropout probability
- Increase weight decay coefficient
- Reduce network size
- Add more regularization
- Get more training data

### 3. Model is underfitting, how can I fix it?

- Increase network capacity (more layers or more neurons per layer)
- Reduce regularization strength
- Increase training epochs
- Try more complex activation functions (GELU)
- Try adaptive optimizers (Adam)

### 4. Loss becomes NaN, how to handle it?

- Lower the learning rate
- Check batch normalization implementation
- Verify data preprocessing is correct
- Use gradient clipping
- Check weight initialization

### 5. How to combine batch normalization and Dropout?

Best practice is to apply batch normalization after activation function and before Dropout:

1. Linear transformation
2. Batch normalization
3. Activation function
4. Dropout

## Appendix: Complete API Reference

### NeuralNetwork Class

```python
class NeuralNetwork:
    def __init__(self, input_size=128, hidden_sizes=[256, 128], output_size=10,
                 use_bn=True, dropout_prob=0.5, activation='relu')

    def activate(self, x)
    def activate_grad(self, x)
    def relu(self, x)
    def relu_grad(self, x)
    def softmax(self, x)
    def batch_norm_forward(self, x, gamma, beta, running_mean, running_var, training, momentum=0.9)
    def forward(self, X, training=True)
    def compute_loss(self, probs, y)
    def backward(self, X, y)
    def update(self, grads, lr, momentum, weight_decay)
```

### AdamOptimizer Class

```python
class AdamOptimizer:
    def __init__(self, learning_rate=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8)
    def initialize(self, layers)
    def update(self, layers, grads)
```

### DataPreprocessor Class

```python
class DataPreprocessor:
    def __init__(self, method='standard', n_components=None)
    def mean_(self)
    def std_(self)
    def min_(self)
    def scale_(self)
    def fit(self, X)
    def transform(self, X)
    def fit_transform(self, X)
    def inverse_transform(self, X)
    def get_explained_variance_ratio(self)
    def get_params(self)
```

### ModelEvaluator Class

```python
class ModelEvaluator:
    def __init__(self, log_dir='logs')
    def evaluate(self, model, X_test, y_test, model_name="default", features=None, verbose=True)
    def visualize_confusion_matrix(self, cm, class_names=None)
    def _save_metrics(self, metrics, confusion_matrix)
    @staticmethod
    def compare_models(log_dir='logs')
```
