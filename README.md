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
- `advanced_ops.py`: Advanced operations (GELU, Adam)
- `visualize.py`: Visualization utilities
- `run_ablation_studies.py`: Script to run multiple model configurations
- `Analysis.md`: Detailed explanation of all model components

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
python3 main.py --preprocess standard --use_gelu --use_adam

# Min-max scaling with GELU activation
python3 main.py --preprocess minmax --use_gelu
```

### Available Options

Preprocessing:

- `--preprocess standard`: Use standard scaling (mean=0, std=1)
- `--preprocess minmax`: Use min-max scaling (range [0,1])

Features:

- `--use_gelu`: Use GELU activation instead of ReLU
- `--use_adam`: Use Adam optimizer instead of SGD

### Customizing Hyperparameters

```bash
python3 main.py --preprocess standard --use_gelu --use_adam \
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

## Running Ablation Studies

The project includes a script to automatically run multiple model configurations to analyze the impact of individual components:

```bash
python3 run_ablation_studies.py
```

This script will:

1. Create a timestamped directory in the logs folder (e.g., logs/ablation_20250412)
2. Run experiments with various configurations:
   - Baseline model with standard scaling
   - Baseline model with min-max scaling
   - Individual feature tests (GELU, Adam)
   - Feature combinations
   - Architecture variations
   - Learning rate variations
   - Dropout variations
3. Save results in separate subdirectories for each experiment
4. Generate a comprehensive summary of all experiments

Each experiment has a timeout of 30 minutes to prevent hanging. Failed experiments are logged and reported in the summary.

### Viewing Ablation Results

After running the ablation studies:

```bash
python3 -c "from evaluate import ModelEvaluator; ModelEvaluator.compare_models('logs/ablation_YYYYMMDD')"
```

This will show a comparison table with:

- Model name and features
- Preprocessing method used
- Accuracy and F1 score
- Hyperparameters (dropout, learning rate, batch size, architecture)
- Training configuration

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
  │
  └── ablation_YYYYMMDD/      # Ablation studies
      ├── standard_baseline/
      ├── minmax_baseline/
      ├── standard_gelu/
      ├── minmax_adam/
      └── ...
```

Each experiment directory contains complete metrics, history, and visualizations.

## Hardware and Software Requirements

- Python 3.6+
- NumPy, Matplotlib, scikit-learn, tqdm, psutil
- Recommended: CPU with 4+ cores, 8GB+ RAM
