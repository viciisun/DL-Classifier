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
- Learning rate scheduling (step, cosine, exponential)
- Early stopping
- Data preprocessing (standard scaling, min-max scaling, PCA)
- Comprehensive evaluation metrics (accuracy, precision, recall, F1-score)
- Confusion matrix visualization
- Logging and model comparison

## Requirements

```
numpy
matplotlib
scikit-learn
tqdm
```

Install dependencies with:

```bash
pip install numpy matplotlib scikit-learn tqdm
```

## Project Structure

- `main.py`: Main script to run the model
- `model.py`: Neural network implementation
- `train.py`: Training functionality
- `evaluate.py`: Evaluation metrics and visualization
- `data_loader.py`: Data loading and initial preprocessing
- `preprocess.py`: Advanced preprocessing techniques
- `advanced_ops.py`: Advanced operations (GELU, Adam)
- `visualize.py`: Visualization utilities
- `run_ablation_studies.py`: Script to run multiple model configurations
- `Analysis.md`: Detailed explanation of all model components

## Usage

### Basic Usage

To run the default model (2-layer neural network with ReLU, batch normalization, dropout, and SGD with momentum):

```bash
python3 main.py
```

### Using Advanced Features

To run the model with specific advanced features:

```bash
python3 main.py preprocess GELU Adam
```

This will use the default model with preprocessing, GELU activation, and Adam optimizer.

### Running the Full Model

To run the model with all advanced features:

```bash
python3 main.py full
```

This includes preprocessing, GELU activation, Adam optimizer, learning rate scheduling, and early stopping.

### Customizing Hyperparameters

```bash
python3 main.py full --hidden_sizes 512 256 128 --epochs 200 --batch_size 64 --lr 0.0005
```

### Available Command Line Arguments

- `features`: List of features to use (preprocess, GELU, Adam, lr_scheduler, early_stopping, or full)
- `--hidden_sizes`: List of hidden layer sizes (default: [256, 128])
- `--epochs`: Number of training epochs (default: 100)
- `--batch_size`: Mini-batch size (default: 128)
- `--lr`: Learning rate (default: 0.001)
- `--momentum`: Momentum coefficient for SGD (default: 0.9)
- `--weight_decay`: Weight decay coefficient (default: 1e-4)
- `--dropout`: Dropout probability (default: 0.5)
- `--patience`: Patience for early stopping (default: 10)
- `--preprocess_method`: Preprocessing method (standard, minmax, pca) (default: standard)
- `--pca_components`: Number of PCA components (default: None)
- `--scheduler_type`: Learning rate scheduler type (step, cosine, exponential) (default: cosine)
- `--log_dir`: Directory to save logs (default: logs)
- `--verbose`: Print verbose output

## Running Ablation Studies

The project includes a script to automatically run multiple model configurations to analyze the impact of individual components. This helps in understanding which features contribute most to model performance.

### To run ablation studies:

```bash
python3 run_ablation_studies.py
```

This script will:

1. Create a timestamped directory in the logs folder (e.g., logs/ablation_20230501_123456)
2. Run 15 different experiments with various configurations:
   - Baseline model (default features)
   - Models with individual features (GELU, Adam, preprocessing)
   - Models with different preprocessing methods
   - Models with combinations of features
   - Models with different architectures
   - Models with different learning rates
   - Full model with all features
3. Save all results in the same directory for easy comparison

The complete process may take significant time as it's training multiple models sequentially.

### To compare all models after ablation studies:

```bash
python3 -c "from evaluate import ModelEvaluator; ModelEvaluator.compare_models('logs/ablation_YOUR_TIMESTAMP')"
```

Replace `YOUR_TIMESTAMP` with the timestamp from your ablation study run.

## Example Commands for Individual Ablation Studies

### Baseline Model

```bash
python3 main.py
```

### Effect of GELU Activation

```bash
python3 main.py GELU
```

### Effect of Adam Optimizer

```bash
python3 main.py Adam
```

### Effect of Preprocessing

```bash
python3 main.py preprocess --preprocess_method standard
python3 main.py preprocess --preprocess_method minmax
python3 main.py preprocess --preprocess_method pca --pca_components 64
```

### Combined Effects

```bash
python3 main.py GELU Adam
python3 main.py GELU preprocess
python3 main.py Adam preprocess
```

## Viewing Results

After running the model, you can find the logs and results in the `logs` directory. Each run creates a timestamped subdirectory containing:

- Training history
- Evaluation metrics
- Confusion matrix visualization
- Model summary

## Hardware and Software Requirements

- Python 3.6+
- NumPy, Matplotlib, scikit-learn, tqdm
- Recommended: CPU with 4+ cores, 8GB+ RAM
