#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ablation Study and Hyperparameter Analysis Automation Script

This script automatically runs a series of model configurations for:
1. Ablation Studies - Systematically removing or replacing model components
2. Hyperparameter Analysis - Exploring the effects of different hyperparameter settings

Usage:
    python3 run_ablation_study.py 
    
Optional arguments:
    --type {ablation,hyperparams,all}: Type of study to run (default: all)
    --epochs: Training epochs per experiment (default: 30)
    --timeout: Timeout per experiment in minutes (default: 30)
    --log_dir: Log directory (default: logs/study_{date}_{sequence})
    --model_config: Model configuration to use for hyperparameter analysis (format: "activation,optimizer,normalization")
    --fixed_lr: Fixed learning rate for hyperparameter analysis
    --fixed_bs: Fixed batch size for hyperparameter analysis
    --fixed_wd: Fixed weight decay for hyperparameter analysis
    --fixed_dropout: Fixed dropout rate for hyperparameter analysis
    --fixed_hidden: Fixed hidden layer sizes (format: "64 32" for two layers)
    --skip_params: Parameters to skip in analysis (comma-separated: "lr,bs,wd,dropout,hidden")
"""

import os
import argparse
import subprocess
import time
import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import signal
import sys
import glob
from evaluate import ModelEvaluator

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Experiment timeout")

def get_next_sequence_number(base_dir="logs"):
    """Get the next sequence number for study folders"""
    date_str = datetime.now().strftime("%Y%m%d")
    pattern = os.path.join(base_dir, f"study_{date_str}_*")
    existing_dirs = glob.glob(pattern)
    
    if not existing_dirs:
        return 1
    
    # Extract sequence numbers from existing directories
    seq_numbers = []
    for dir_path in existing_dirs:
        try:
            seq_part = dir_path.split('_')[-1]
            seq_numbers.append(int(seq_part))
        except (ValueError, IndexError):
            continue
    
    # Return the next sequence number
    return max(seq_numbers, default=0) + 1

def run_experiment(command, timeout_minutes=30):
    """Run a single experiment with timeout mechanism"""
    start_time = time.time()
    print(f"\n{'='*50}")
    print(f"Executing command: {command}")
    print(f"{'='*50}")
    
    # Set timeout signal handler
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(int(timeout_minutes * 60))  # Convert to seconds
    
    try:
        process = subprocess.run(command, shell=True, check=True, 
                                stderr=subprocess.STDOUT, text=True)
        status = "Success"
    except subprocess.CalledProcessError as e:
        print(f"Experiment failed with error code {e.returncode}")
        status = f"Failed (code {e.returncode})"
    except TimeoutError:
        print(f"Experiment timed out (>{timeout_minutes} minutes)")
        status = f"Timeout (>{timeout_minutes} minutes)"
    
    # Cancel timeout alarm
    signal.alarm(0)
    
    end_time = time.time()
    runtime = end_time - start_time
    print(f"Runtime: {runtime:.2f} seconds")
    print(f"Status: {status}")
    
    return {
        "command": command,
        "status": status,
        "runtime": runtime
    }

def run_ablation_studies(epochs, log_dir, timeout):
    """Run ablation studies"""
    print("\n\n====== Starting Ablation Studies ======\n")
    
    ablation_dir = os.path.join(log_dir, "ablation")
    os.makedirs(ablation_dir, exist_ok=True)
    
    experiments = []
    
    # Baseline model - full features with best configuration
    # Using a strong baseline with optimized hyperparameters
    baseline_cmd = f"python3 main.py --preprocess standard --gelu --adam --hidden_sizes 512 256 --lr 0.0005 --batch_size 128 --dropout 0.5 --weight_decay 0.0001 --epochs {epochs} --log_dir {ablation_dir}"
    
    experiments.append({
        "name": "baseline_full_model",
        "command": baseline_cmd,
        "type": "baseline"
    })
    
    # Systematically remove or alter components
    ablation_experiments = [
        # 1. Activation functions
        {
            "name": "no_gelu_use_relu",
            "command": baseline_cmd.replace(" --gelu", ""),
            "type": "activation"
        },
        # 2. Optimizers
        {
            "name": "no_adam_use_sgd",
            "command": baseline_cmd.replace(" --adam", ""),
            "type": "optimizer"
        },
        # 3. Normalization
        {
            "name": "no_batch_norm",
            "command": f"{baseline_cmd} --no_batch_norm",
            "type": "normalization"
        },
        # 4. Preprocessing
        {
            "name": "minmax_scaling",
            "command": baseline_cmd.replace("--preprocess standard", "--preprocess minmax"),
            "type": "preprocessing"
        },
        {
            "name": "robust_scaling",
            "command": baseline_cmd.replace("--preprocess standard", "--preprocess robust"),
            "type": "preprocessing"
        },
        # 5. Network width
        {
            "name": "narrow_network",
            "command": baseline_cmd.replace("--hidden_sizes 512 256", "--hidden_sizes 128 64"),
            "type": "architecture"
        },
        # 6. Network depth
        {
            "name": "shallow_network",
            "command": baseline_cmd.replace("--hidden_sizes 512 256", "--hidden_sizes 256"),
            "type": "architecture"
        },
        {
            "name": "deep_network",
            "command": baseline_cmd.replace("--hidden_sizes 512 256", "--hidden_sizes 512 256 128 64"),
            "type": "architecture"
        },
        # 7. Regularization
        {
            "name": "no_dropout",
            "command": baseline_cmd.replace("--dropout 0.5", "--dropout 0.0"),
            "type": "regularization"
        },
        {
            "name": "no_weight_decay",
            "command": baseline_cmd.replace("--weight_decay 0.0001", "--weight_decay 0"),
            "type": "regularization"
        },
        # 8. Combined ablations (for interaction effects)
        {
            "name": "no_regularization",
            "command": baseline_cmd.replace("--dropout 0.5", "--dropout 0.0").replace("--weight_decay 0.0001", "--weight_decay 0"),
            "type": "combined"
        },
        {
            "name": "minimal_model",
            "command": ("python3 main.py --preprocess standard --hidden_sizes 256 "
                       f"--lr 0.001 --batch_size 128 --dropout 0.0 --weight_decay 0 --epochs {epochs} --log_dir {ablation_dir}"),
            "type": "combined"
        }
    ]
    
    experiments.extend(ablation_experiments)
    
    # Execute experiments
    results = []
    print(f"Total ablation experiments planned: {len(experiments)}")
    
    for i, exp in enumerate(experiments):
        print(f"\nRunning ablation experiment {i+1}/{len(experiments)}: {exp['name']} ({exp['type']})")
        exp_result = run_experiment(exp["command"], timeout)
        exp_result["name"] = exp["name"]
        exp_result["type"] = exp["type"]
        results.append(exp_result)
    
    # Save result summary
    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "baseline": baseline_cmd,
        "experiments": results
    }
    
    with open(os.path.join(ablation_dir, "ablation_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)
    
    print("\n====== Ablation Studies Completed ======\n")
    return results

def run_hyperparameter_analysis(epochs, log_dir, timeout, model_config=None, fixed_params=None, skip_params=None):
    """Run hyperparameter analysis"""
    print("\n\n====== Starting Hyperparameter Analysis ======\n")
    
    hyperparams_dir = os.path.join(log_dir, "hyperparams")
    os.makedirs(hyperparams_dir, exist_ok=True)
    
    # Parse model configuration if provided
    activation = "gelu"
    optimizer = "adam"
    batch_norm = True
    
    if model_config:
        config_parts = model_config.split(',')
        if len(config_parts) >= 1 and config_parts[0]:
            activation = config_parts[0]
        if len(config_parts) >= 2 and config_parts[1]:
            optimizer = config_parts[1]
        if len(config_parts) >= 3 and config_parts[2] == "no_batch_norm":
            batch_norm = False
    
    # Initialize fixed parameters and skip parameters sets
    if fixed_params is None:
        fixed_params = {}
    
    if skip_params is None:
        skip_params = set()
    
    # Build base command according to configuration
    base_cmd = f"python3 main.py --preprocess standard"
    
    if activation == "gelu":
        base_cmd += " --gelu"
    # No need for ReLU flag as it's the default
    
    if optimizer == "adam":
        base_cmd += " --adam"
    # No need for SGD flag as it's the default
    
    if not batch_norm:
        base_cmd += " --no_batch_norm"
    
    print(f"Using base configuration: {base_cmd}")
    print(f"Fixed parameters: {fixed_params}")
    print(f"Skipped parameters: {skip_params}")
    
    experiments = []
    
    # Define parameter ranges
    param_ranges = {}
    
    if 'lr' not in skip_params and 'lr' not in fixed_params:
        param_ranges['lr'] = [0.0001, 0.0005, 0.001, 0.005]
    else:
        print("Skipping learning rate analysis")
        
    if 'bs' not in skip_params and 'bs' not in fixed_params:
        param_ranges['bs'] = [32, 64, 128, 256]
    else:
        print("Skipping batch size analysis")
        
    if 'wd' not in skip_params and 'wd' not in fixed_params:
        param_ranges['wd'] = [0, 1e-5, 0.0001, 0.001]
    else:
        print("Skipping weight decay analysis")
        
    if 'dropout' not in skip_params and 'dropout' not in fixed_params:
        param_ranges['dropout'] = [0.0, 0.3, 0.5, 0.7]
    else:
        print("Skipping dropout analysis")
        
    if 'hidden' not in skip_params and 'hidden' not in fixed_params:
        param_ranges['hidden'] = ["256 128", "512 256", "256 128 64", "512 256 128", "128 64"]
    else:
        print("Skipping hidden layer size analysis")
    
    # 1. First run single parameter experiments
    for param, values in param_ranges.items():
        for value in values:
            cmd = base_cmd
            name_parts = []
            
            # Add fixed parameters to command
            for fixed_param, fixed_value in fixed_params.items():
                if fixed_param == 'hidden':
                    cmd += f" --hidden_sizes {fixed_value}"
                elif fixed_param == 'lr':
                    cmd += f" --lr {fixed_value}"
                elif fixed_param == 'bs':
                    cmd += f" --batch_size {fixed_value}"
                elif fixed_param == 'wd':
                    cmd += f" --weight_decay {fixed_value}"
                elif fixed_param == 'dropout':
                    cmd += f" --dropout {fixed_value}"
                name_parts.append(f"{fixed_param}_{fixed_value}")
            
            # Add current parameter being tested
            if param == 'hidden':
                cmd += f" --hidden_sizes {value}"
                name_parts.append(f"hidden_{value.replace(' ', '_')}")
            elif param == 'lr':
                cmd += f" --lr {value}"
                name_parts.append(f"lr_{value}")
            elif param == 'bs':
                cmd += f" --batch_size {value}"
                name_parts.append(f"bs_{value}")
            elif param == 'wd':
                cmd += f" --weight_decay {value}"
                name_parts.append(f"wd_{value}")
            elif param == 'dropout':
                cmd += f" --dropout {value}"
                name_parts.append(f"dropout_{value}")
            
            name = "_".join(name_parts) if name_parts else f"{param}_{value}"
            experiments.append({
                "name": name,
                "command": f"{cmd} --epochs {epochs} --log_dir {hyperparams_dir}",
                "type": "single_param"
            })
    
    # 2. Then run focused grid search with top parameters
    # Focus on combinations of batch size and hidden layers
    if 'bs' in param_ranges and 'hidden' in param_ranges:
        print("Running focused grid search for batch size and hidden layers...")
        
        # Select a subset of values for efficiency
        bs_values = param_ranges['bs'][:3]  # First 3 batch sizes
        hidden_values = param_ranges['hidden'][:3]  # First 3 hidden layer configs
        
        for bs in bs_values:
            for hidden in hidden_values:
                cmd = base_cmd
                name_parts = [f"grid_bs_{bs}_hidden_{hidden.replace(' ', '_')}"]
                
                # Add fixed parameters
                for fixed_param, fixed_value in fixed_params.items():
                    if fixed_param not in ['bs', 'hidden']:  # Skip if we're already varying these
                        if fixed_param == 'lr':
                            cmd += f" --lr {fixed_value}"
                        elif fixed_param == 'wd':
                            cmd += f" --weight_decay {fixed_value}"
                        elif fixed_param == 'dropout':
                            cmd += f" --dropout {fixed_value}"
                        name_parts.append(f"{fixed_param}_{fixed_value}")
                
                # Add current batch size and hidden sizes being tested
                cmd += f" --batch_size {bs} --hidden_sizes {hidden}"
                
                name = "_".join(name_parts)
                experiments.append({
                    "name": name,
                    "command": f"{cmd} --epochs {epochs} --log_dir {hyperparams_dir}",
                    "type": "grid_search"
                })
    
    # 3. Add top configurations based on common good values
    # These combinations often work well together
    top_configs = [
        {"lr": 0.0005, "bs": 128, "dropout": 0.5, "hidden": "512 256", "wd": 0.0001},
        {"lr": 0.0005, "bs": 64, "dropout": 0.5, "hidden": "512 256", "wd": 0.0001},
        {"lr": 0.0005, "bs": 128, "dropout": 0.3, "hidden": "256 128", "wd": 0.0001},
        {"lr": 0.0005, "bs": 64, "dropout": 0.3, "hidden": "256 128", "wd": 0.0001},
    ]
    
    for i, config in enumerate(top_configs):
        cmd = base_cmd
        name_parts = [f"top_config_{i+1}"]
        
        # Apply fixed parameters (overriding top configs)
        for param in ['lr', 'bs', 'dropout', 'hidden', 'wd']:
            if param in fixed_params:
                config[param] = fixed_params[param]
        
        # Skip if any parameter should be skipped
        skip_config = False
        for param in config:
            if param in skip_params:
                skip_config = True
                break
        
        if skip_config:
            continue
        
        # Add configuration to command
        cmd += f" --lr {config['lr']} --batch_size {config['bs']} --dropout {config['dropout']} --hidden_sizes {config['hidden']} --weight_decay {config['wd']}"
        
        name = "_".join(name_parts)
        experiments.append({
            "name": name,
            "command": f"{cmd} --epochs {epochs} --log_dir {hyperparams_dir}",
            "type": "top_config"
        })
    
    # Execute experiments
    results = []
    print(f"Total experiments planned: {len(experiments)}")
    for i, exp in enumerate(experiments):
        print(f"\nRunning experiment {i+1}/{len(experiments)}: {exp['name']} ({exp['type']})")
        exp_result = run_experiment(exp["command"], timeout)
        exp_result["name"] = exp["name"]
        exp_result["type"] = exp["type"]
        results.append(exp_result)
    
    # Save result summary
    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_config": model_config,
        "fixed_params": fixed_params,
        "skip_params": list(skip_params),
        "experiments": results
    }
    
    with open(os.path.join(hyperparams_dir, "hyperparams_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)
    
    print("\n====== Hyperparameter Analysis Completed ======\n")
    return results

def generate_report(log_dir):
    """Generate experiment report"""
    print("\n\n====== Generating Experiment Report ======\n")
    
    # Compare all models using ModelEvaluator
    try:
        all_models = ModelEvaluator.compare_models(log_dir)
        print(f"Compared {len(all_models)} models")
        
        # Create report directory
        report_dir = os.path.join(log_dir, "reports")
        os.makedirs(report_dir, exist_ok=True)
        
        # Save as CSV
        models_df = pd.DataFrame(all_models)
        csv_path = os.path.join(report_dir, "model_comparison.csv")
        models_df.to_csv(csv_path, index=False)
        print(f"Saved model comparison CSV to {csv_path}")
        
        # Sort models by accuracy for better visualization
        models_df = models_df.sort_values(by='accuracy', ascending=False)
        
        # Create accuracy comparison chart
        plt.figure(figsize=(14, 8))
        bar = plt.bar(models_df['model_name'], models_df['accuracy'])
        plt.title('Model Accuracy Comparison')
        plt.xlabel('Model')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=90, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels on top of bars
        for rect in bar:
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width()/2., height + 0.005,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        accuracy_plot_path = os.path.join(report_dir, "accuracy_comparison.png")
        plt.savefig(accuracy_plot_path)
        plt.close()
        print(f"Saved accuracy comparison chart to {accuracy_plot_path}")
        
        # Add F1 score comparison chart
        if 'f1_score' in models_df.columns:
            plt.figure(figsize=(14, 8))
            bar = plt.bar(models_df['model_name'], models_df['f1_score'])
            plt.title('Model F1 Score Comparison')
            plt.xlabel('Model')
            plt.ylabel('F1 Score')
            plt.xticks(rotation=90, ha='right')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Add value labels on top of bars
            for rect in bar:
                height = rect.get_height()
                plt.text(rect.get_x() + rect.get_width()/2., height + 0.005,
                        f'{height:.4f}', ha='center', va='bottom', fontsize=8)
            
            plt.tight_layout()
            f1_plot_path = os.path.join(report_dir, "f1_comparison.png")
            plt.savefig(f1_plot_path)
            plt.close()
            print(f"Saved F1 score comparison chart to {f1_plot_path}")
        
        # Generate additional analysis - hyperparameter impact
        try:
            # Extract hyperparameter values from model names and configurations
            hyperparams = {}
            
            # Look for batch size patterns
            bs_values = []
            bs_accuracies = []
            for model in all_models:
                if 'batch_size' in model or 'bs_' in model.get('model_name', ''):
                    try:
                        if 'batch_size' in model:
                            bs = int(model['batch_size'])
                        else:
                            # Try to extract from name
                            name = model['model_name']
                            bs_part = [p for p in name.split('_') if p.startswith('bs')]
                            if bs_part:
                                bs = int(bs_part[0].replace('bs', '').replace('_', ''))
                            else:
                                continue
                        
                        bs_values.append(bs)
                        bs_accuracies.append(model['accuracy'])
                    except (ValueError, KeyError):
                        continue
            
            if bs_values:
                hyperparams['batch_size'] = (bs_values, bs_accuracies)
            
            # Generate plots for each hyperparameter if we have data
            for param, (values, accuracies) in hyperparams.items():
                if len(values) > 1:  # Only plot if we have multiple values
                    plt.figure(figsize=(10, 6))
                    
                    # Sort by parameter value
                    sorted_data = sorted(zip(values, accuracies))
                    sorted_values, sorted_accuracies = zip(*sorted_data)
                    
                    plt.plot(sorted_values, sorted_accuracies, 'o-', linewidth=2, markersize=8)
                    plt.title(f'Impact of {param} on Accuracy')
                    plt.xlabel(param)
                    plt.ylabel('Accuracy')
                    plt.grid(True, linestyle='--', alpha=0.7)
                    
                    for x, y in zip(sorted_values, sorted_accuracies):
                        plt.text(x, y + 0.005, f'{y:.4f}', ha='center', va='bottom')
                    
                    plt.tight_layout()
                    param_plot_path = os.path.join(report_dir, f"{param}_impact.png")
                    plt.savefig(param_plot_path)
                    plt.close()
                    print(f"Saved {param} impact chart to {param_plot_path}")
        
        except Exception as e:
            print(f"Error generating hyperparameter impact analysis: {e}")
        
        # Generate a summary text file
        try:
            with open(os.path.join(report_dir, "summary.txt"), "w") as f:
                f.write("=== MODEL COMPARISON SUMMARY ===\n\n")
                f.write(f"Total models evaluated: {len(all_models)}\n")
                f.write("\nTOP 5 PERFORMING MODELS:\n")
                
                for i, model in enumerate(models_df.head(5).to_dict('records')):
                    f.write(f"\n{i+1}. {model['model_name']}\n")
                    f.write(f"   Accuracy: {model['accuracy']:.4f}\n")
                    if 'f1_score' in model:
                        f.write(f"   F1 Score: {model['f1_score']:.4f}\n")
                    
                    # Write model configuration details if available
                    config_keys = [k for k in model.keys() if k not in ['model_name', 'accuracy', 'f1_score']]
                    if config_keys:
                        f.write("   Configuration:\n")
                        for key in config_keys:
                            f.write(f"     - {key}: {model[key]}\n")
                
                f.write("\n=== OBSERVATIONS ===\n")
                f.write("The report shows performance variations across different model configurations.\n")
                f.write("See the generated charts for visual comparison of results.\n")
        
        except Exception as e:
            print(f"Error generating summary text: {e}")
        
    except Exception as e:
        print(f"Error generating report: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Run ablation studies and hyperparameter analysis')
    parser.add_argument('--type', type=str, choices=['ablation', 'hyperparams', 'all'], 
                        default='all', help='Type of study to run')
    parser.add_argument('--epochs', type=int, default=30, 
                        help='Training epochs per experiment')
    parser.add_argument('--timeout', type=int, default=30, 
                        help='Timeout per experiment in minutes')
    parser.add_argument('--log_dir', type=str, default=None, 
                        help='Log directory')
    parser.add_argument('--model_config', type=str, default=None,
                       help='Model configuration for hyperparameter analysis (format: "activation,optimizer,normalization")')
    
    # Fixed parameters
    parser.add_argument('--fixed_lr', type=float, default=None,
                       help='Fixed learning rate for hyperparameter analysis')
    parser.add_argument('--fixed_bs', type=int, default=None,
                       help='Fixed batch size for hyperparameter analysis')
    parser.add_argument('--fixed_wd', type=float, default=None,
                       help='Fixed weight decay for hyperparameter analysis')
    parser.add_argument('--fixed_dropout', type=float, default=None,
                       help='Fixed dropout rate for hyperparameter analysis')
    parser.add_argument('--fixed_hidden', type=str, default=None,
                       help='Fixed hidden layer sizes (format: "64 32" for two layers)')
    
    # Skip parameters
    parser.add_argument('--skip_params', type=str, default=None,
                       help='Parameters to skip in analysis (comma-separated: "lr,bs,wd,dropout,hidden")')
    
    # New options for controlling grid search
    parser.add_argument('--grid_search', action='store_true',
                       help='Enable grid search in hyperparameter analysis')
    parser.add_argument('--top_configs', action='store_true',
                       help='Run predefined top configurations')
    parser.add_argument('--max_experiments', type=int, default=None,
                       help='Maximum number of experiments to run')
    
    args = parser.parse_args()
    
    # Create log directory with sequence number
    if args.log_dir is None:
        date_str = datetime.now().strftime("%Y%m%d")
        seq_num = get_next_sequence_number()
        args.log_dir = f"logs/study_{date_str}_{seq_num:03d}"
    
    os.makedirs(args.log_dir, exist_ok=True)
    print(f"Logs will be saved to: {args.log_dir}")
    
    # Collect fixed parameters
    fixed_params = {}
    if args.fixed_lr is not None:
        fixed_params['lr'] = args.fixed_lr
    if args.fixed_bs is not None:
        fixed_params['bs'] = args.fixed_bs
    if args.fixed_wd is not None:
        fixed_params['wd'] = args.fixed_wd
    if args.fixed_dropout is not None:
        fixed_params['dropout'] = args.fixed_dropout
    if args.fixed_hidden is not None:
        fixed_params['hidden'] = args.fixed_hidden
    
    # Parse skip parameters
    skip_params = set()
    if args.skip_params:
        skip_params = set(args.skip_params.split(','))
    
    # Save run configuration
    config = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "study_type": args.type,
        "epochs": args.epochs,
        "timeout": args.timeout,
        "log_dir": args.log_dir,
        "model_config": args.model_config,
        "fixed_params": fixed_params,
        "skip_params": list(skip_params),
        "grid_search": args.grid_search,
        "top_configs": args.top_configs,
        "max_experiments": args.max_experiments
    }
    
    with open(os.path.join(args.log_dir, "study_config.json"), "w") as f:
        json.dump(config, f, indent=4)
    
    print("\n" + "="*50)
    print(" EXPERIMENT CONFIGURATION ".center(50, "="))
    print("="*50)
    print(f"Study type:       {args.type}")
    print(f"Epochs:           {args.epochs}")
    print(f"Timeout:          {args.timeout} minutes per experiment")
    print(f"Fixed parameters: {fixed_params}")
    print(f"Skipped params:   {skip_params}")
    print(f"Grid search:      {'Enabled' if args.grid_search else 'Disabled'}")
    print(f"Top configs:      {'Enabled' if args.top_configs else 'Disabled'}")
    if args.max_experiments:
        print(f"Max experiments:  {args.max_experiments}")
    print("="*50 + "\n")
    
    # Run studies
    all_results = []
    
    if args.type in ['ablation', 'all']:
        ablation_results = run_ablation_studies(args.epochs, args.log_dir, args.timeout)
        all_results.extend(ablation_results)
    
    if args.type in ['hyperparams', 'all']:
        hyperparams_results = run_hyperparameter_analysis(
            epochs=args.epochs, 
            log_dir=args.log_dir, 
            timeout=args.timeout,
            model_config=args.model_config, 
            fixed_params=fixed_params, 
            skip_params=skip_params
        )
        all_results.extend(hyperparams_results)
    
    # Generate report
    generate_report(args.log_dir)
    
    print("\n" + "="*50)
    print(" EXPERIMENT SUMMARY ".center(50, "="))
    print("="*50)
    
    # Print brief summary
    success_count = sum(1 for r in all_results if r["status"] == "Success")
    print(f"Ran a total of {len(all_results)} experiments")
    print(f"Successful:      {success_count}")
    print(f"Failed:          {len(all_results) - success_count}")
    
    # Calculate average runtime
    runtimes = [r["runtime"] for r in all_results if r["status"] == "Success"]
    if runtimes:
        avg_runtime = sum(runtimes) / len(runtimes)
        print(f"Average runtime: {avg_runtime:.2f} seconds")
    
    print("="*50)
    
    if success_count > 0:
        print(f"\nTo view detailed results, run:")
        print(f"python3 -c \"from evaluate import ModelEvaluator; ModelEvaluator.compare_models('{args.log_dir}')\"")
        print(f"\nReports saved to: {os.path.join(args.log_dir, 'reports')}")

if __name__ == "__main__":
    main() 