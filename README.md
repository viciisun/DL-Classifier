# Deep Learning Classifier

A comprehensive implementation of a neural network classifier from scratch for COMP4329 Deep Learning assignment.

## Features

### Basic Features

- Multi-layer neural network with customizable architecture
- ReLU activation functions
- Weight decay (L2 regularization)
- SGD with momentum
- Dropout regularization
- Softmax activation and cross-entropy loss
- Mini-batch training
- Batch Normalization

### Advanced Features

- GELU activation function
- Adam optimizer
- Early stopping
- Data preprocessing options:
  - Standard scaling (zero mean, unit variance)
  - Min-max scaling (range [0,1])
- Comprehensive evaluation metrics (accuracy, precision, recall, F1-score)
- Confusion matrix visualization
- Structured logging and model comparison

## Requirements

```
numpy
matplotlib
scikit-learn
tqdm
psutil
```

Install dependencies with:

```bash
pip install numpy matplotlib scikit-learn tqdm psutil
```

## Project Structure

- `main.py`: Main script to run the model
- `cli.py`: Command line interface and argument parsing
- `model.py`: Neural network implementation
- `train.py`: Training functionality
- `evaluate.py`: Evaluation metrics and visualization
- `data_loader.py`: Data loading and initial preprocessing
- `preprocess.py`: Data preprocessing (standard and min-max scaling)
- `visualize.py`: Visualization utilities
- `Analysis.md`: Detailed explanation of all model components

## Dataset

The project uses MNIST dataset in NumPy format:

- `data/train_data.npy`: Training data features
- `data/train_label.npy`: Training data labels
- `data/test_data.npy`: Test data features
- `data/test_label.npy`: Test data labels

## Model Implementation

The neural network classifier is implemented from scratch using NumPy, with the following key components:

1. **Forward Propagation**: Computes predictions in a layer-by-layer fashion
2. **Backpropagation**: Calculates gradients for all parameters using the chain rule
3. **Optimizers**: Implements SGD with momentum and Adam optimizer
4. **Regularization**: Includes dropout, batch normalization, and L2 regularization
5. **Activation Functions**: Provides ReLU and GELU activation functions

## Usage

### Basic Usage

To run the model with standard scaling:

```bash
python3 main.py --preprocess standard
```

To run the model with min-max scaling:

```bash
python3 main.py --preprocess minmax
```

### Using Advanced Features

To run the model with specific advanced features:

```bash
# Standard scaling with GELU activation and Adam optimizer
python3 main.py --preprocess standard --gelu --adam

# Min-max scaling with GELU activation
python3 main.py --preprocess minmax --gelu
```

### Customizing Hyperparameters

```bash
python3 main.py --preprocess standard --gelu --adam \
                --hidden_sizes 512 256 128 \
                --epochs 200 \
                --batch_size 64 \
                --lr 0.0005
```

### Available Command Line Arguments

Model Architecture:

- `--hidden_sizes`: List of hidden layer sizes (default: [256, 128])

Training Parameters:

- `--epochs`: Number of training epochs (default: 100)
- `--batch_size`: Mini-batch size (default: 128)
- `--lr`: Learning rate (default: 0.001)
- `--momentum`: Momentum coefficient for SGD (default: 0.9)
- `--weight_decay`: Weight decay coefficient (default: 1e-4)
- `--dropout`: Dropout probability (default: 0.5)
- `--early_stopping_patience`: Number of epochs to wait before early stopping (default: 10)

Output Options:

- `--log_dir`: Directory to save logs (default: logs)
- `--model_name`: Custom model name (default: auto-generated based on features)

## Experimental Results

The best model configuration achieves 97.82% accuracy on the test set with the following configuration:

- Preprocessing: Standard scaling
- Activation: GELU
- Optimizer: Adam
- Hidden layers: [256, 128]
- Dropout: 0.5
- Batch Normalization: Enabled

For a detailed analysis of experiments and results, please refer to `Analysis.md`.

## Visualizations

The model generates various visualizations during training and evaluation:

1. **Training History**: Shows the training/validation loss and accuracy over epochs
2. **Confusion Matrix**: Visualizes the model's predictions across different classes
3. **Model Comparison**: Compares different model configurations on key metrics

All visualizations are saved in the logs directory.

## Log Structure

Logs are organized as follows:

```
logs/
  ├── YYYYMMDD_001/           # Regular training runs
  │   ├── system_info.json
  │   ├── model_metrics.json
  │   ├── model_history.json
  │   ├── preprocess_info.json
  │   └── confusion_matrix.png
```

Each experiment directory contains complete metrics, history, and visualizations.

## Hardware and Software Requirements

- Python 3.6+
- NumPy, Matplotlib, scikit-learn, tqdm, psutil
- Recommended: CPU with 4+ cores, 8GB+ RAM

## Authors

- Your Name

## License

This project is available for academic and educational purposes.
