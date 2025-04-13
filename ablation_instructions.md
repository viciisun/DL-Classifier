# Ablation Study and Hyperparameter Analysis Instructions

This document provides instructions for running ablation studies and hyperparameter analysis experiments using the provided scripts. These tools help understand the impact of different components and parameters on model performance.

## Overview

This document provides detailed instructions on how to use the `run_ablation_study.py` script for conducting ablation studies and hyperparameter analysis.

`run_ablation_study.py` is an automated tool for systematically evaluating the impact of different neural network components and hyperparameters on model performance. This script can:

1. Perform **Ablation Studies**: Evaluate the contribution of each component to overall performance by removing or replacing specific model components
2. Conduct **Hyperparameter Analysis**: Test the effects of different hyperparameter settings, such as learning rate, batch size, etc.
3. Generate comparison reports and visual results

## Basic Usage

Run the ablation study and hyperparameter analysis script as follows:

```bash
python3 run_ablation_study.py [options]
```

### Command Line Options

The script supports the following command-line options:

```
--type {ablation,hyperparams,all}   Type of study to run (default: all)
--epochs N                          Training epochs per experiment (default: 30)
--timeout N                         Timeout per experiment in minutes (default: 30)
--log_dir PATH                      Log directory (default: logs/study_{date}_{sequence})
--model_config CONFIG               Model configuration for hyperparameter analysis (format: "activation,optimizer,normalization")
```

### Fixed Parameters Options

You can fix specific hyperparameters with these options:

```
--fixed_lr VALUE                    Fixed learning rate for hyperparameter analysis
--fixed_bs VALUE                    Fixed batch size for hyperparameter analysis
--fixed_wd VALUE                    Fixed weight decay for hyperparameter analysis
--fixed_dropout VALUE               Fixed dropout rate for hyperparameter analysis
--fixed_hidden VALUE                Fixed hidden layer sizes (format: "64 32" for two layers)
--skip_params LIST                  Parameters to skip in analysis (comma-separated: "lr,bs,wd,dropout,hidden")
```

### Usage Examples

1. Run both ablation study and hyperparameter analysis with default settings:

   ```bash
   python3 run_ablation_study.py
   ```

2. Run only ablation study with 50 epochs:

   ```bash
   python3 run_ablation_study.py --type ablation --epochs 50
   ```

3. Run only hyperparameter analysis with GELU activation, Adam optimizer, and no batch normalization:

   ```bash
   python3 run_ablation_study.py --type hyperparams --model_config "gelu,adam,no_batch_norm"
   ```

4. Run hyperparameter analysis with fixed learning rate and batch size:

   ```bash
   python3 run_ablation_study.py --type hyperparams --fixed_lr 0.005 --fixed_bs 128
   ```

5. Run hyperparameter analysis with specific model configuration and fixed parameters:

   ```bash
   python3 run_ablation_study.py --type hyperparams --model_config "gelu,adam,no_batch_norm" --fixed_lr 0.005 --fixed_bs 128 --epochs 100
   ```

6. Run hyperparameter analysis testing only dropout and hidden layers:

   ```bash
   python3 run_ablation_study.py --type hyperparams --skip_params "lr,bs,wd"
   ```

7. Specify custom log directory:
   ```bash
   python3 run_ablation_study.py --log_dir logs/my_custom_study
   ```

## Ablation Study Details

Ablation studies systematically remove or modify various components of the model to evaluate their contribution to final performance. The current implementation includes the following analyses:

1. **Activation Function Analysis**: GELU vs. ReLU
2. **Optimizer Analysis**: Adam vs. SGD with Momentum
3. **Batch Normalization Analysis**: Enabled vs. Disabled
4. **Dropout Analysis**: Testing different Dropout probabilities (0.0, 0.3, 0.7)
5. **Preprocessing Method Analysis**: Standardization vs. Min-Max Scaling
6. **Network Architecture Analysis**: Testing different network structures (single layer, wider, deeper)

## Hyperparameter Analysis Details

Hyperparameter analysis explores different hyperparameter settings to find the optimal configuration. The current implementation includes the following analyses:

1. **Learning Rate Analysis**: Testing different learning rates (0.0001, 0.0005, 0.001, 0.005, 0.01)
2. **Batch Size Analysis**: Testing different batch sizes (32, 64, 128, 256)
3. **Weight Decay Analysis**: Testing different L2 regularization strengths (0, 1e-5, 1e-4, 1e-3)
4. **Hidden Layer Size Analysis**: Testing different network configuration structures

## Reports and Results

After completion, the results will be saved in the specified log directory with the following structure:

```
logs/study_YYYYMMDD/
  ├── study_config.json            # Study configuration
  ├── ablation/                    # Ablation study results
  │   ├── ablation_summary.json    # Ablation study summary
  │   └── ...                      # Individual experiment logs
  ├── hyperparams/                 # Hyperparameter analysis results
  │   ├── hyperparams_summary.json # Hyperparameter analysis summary
  │   └── ...                      # Individual experiment logs
  └── reports/                     # Generated reports
      ├── model_comparison.csv     # Model comparison table
      └── accuracy_comparison.png  # Accuracy comparison chart
```

To view a detailed comparison of all models, run:

```bash
python3 -c "from evaluate import ModelEvaluator; ModelEvaluator.compare_models('logs/study_YYYYMMDD')"
```

## Extending and Customizing

### Adding New Ablation Experiments

To add new ablation experiments, edit the `experiments` list in the `run_ablation_studies` function:

```python
experiments.append({
    "name": "my_new_experiment",
    "command": f"python3 main.py --preprocess standard --some_new_flag --epochs {epochs} --log_dir {ablation_dir}"
})
```

### Adding New Hyperparameter Analyses

To add new hyperparameter analyses, edit the corresponding section in the `run_hyperparameter_analysis` function:

```python
# For example, adding new learning rate values
learning_rates = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]  # Added 0.05
```

## Troubleshooting

### Experiment Timeout

If experiments are timing out, you can solve this by increasing the timeout duration:

```bash
python3 run_ablation_study.py --timeout 60  # Increase to 60 minutes
```

### Memory Issues

If you encounter memory issues, try reducing batch sizes or network sizes:

```bash
# For example, modify batch size test range for hyperparameter study
batch_sizes = [16, 32, 64, 128]  # Use smaller batch sizes
```

## References

For more detailed information, please refer to the following resources:

- Project `README.md`: Provides project overview and basic usage
- `manual.md`: Provides detailed project manual
- `Analysis.md`: Contains in-depth analysis of experimental results
