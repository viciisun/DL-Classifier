# Deep Learning Classifier

A comprehensive implementation of a neural network classifier.

## Instruction

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

### Running Automated Experiments

For systematic model evaluation, you can use the `run_experiment.py` script that automates ablation studies and hyperparameter analysis.

#### Ablation Studies

Ablation studies systematically remove or replace model components to understand their impact:

```bash
# Run only ablation studies with 50 epochs per experiment
python3 run_experiment.py --type ablation --epochs 50

# Run ablation studies with custom timeout (in minutes)
python3 run_experiment.py --type ablation --timeout 60
```

#### Hyperparameter Analysis

Explore different hyperparameter configurations:

```bash
# Run only hyperparameter analysis
python3 run_experiment.py --type hyperparams

# Specify base model configuration for hyperparameter analysis
python3 run_experiment.py --type hyperparams --model_config "gelu,adam,batch_norm"

# Fix specific hyperparameters during analysis
python3 run_experiment.py --type hyperparams --fixed_lr 0.001 --fixed_dropout 0.5

# Skip specific hyperparameter groups in analysis
python3 run_experiment.py --type hyperparams --skip_params "lr,dropout"
```

#### Combined Experiments

Run both ablation studies and hyperparameter analysis:

```bash
# Run all experiments with custom log directory
python3 run_experiment.py --type all --log_dir logs/my_experiments
```

#### Available Command Line Arguments for run_experiment.py

- `--type`: Type of study to run (`ablation`, `hyperparams`, or `all`, default: `all`)
- `--epochs`: Training epochs per experiment (default: 30)
- `--timeout`: Timeout per experiment in minutes (default: 30)
- `--log_dir`: Log directory (default: logs/study*{date}*{sequence})
- `--model_config`: Model configuration for hyperparameter analysis (format: "activation,optimizer,normalization")
- `--fixed_lr`: Fixed learning rate for hyperparameter analysis
- `--fixed_bs`: Fixed batch size for hyperparameter analysis
- `--fixed_wd`: Fixed weight decay for hyperparameter analysis
- `--fixed_dropout`: Fixed dropout rate for hyperparameter analysis
- `--fixed_hidden`: Fixed hidden layer sizes (format: "64 32" for two layers)
- `--skip_params`: Parameters to skip in analysis (comma-separated: "lr,bs,wd,dropout,hidden")

### To compare logs results

```bash
python3 compare_models.py logs/destination_folder
```

### Using Shell Scripts

The repository also includes two shell scripts for running experiments:

```bash
# Run full ablation study (24 model configurations)
bash run_ablation.sh

# Run hyperparameter grid search
bash run_hyperparams.sh
```

## Features

### Basic Features

- Multi-layer neural network with customizable architecture
- ReLU activation functions
- Weight decay
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
