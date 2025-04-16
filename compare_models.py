#!/usr/bin/env python3

import os
import json
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

def compare_models(log_dir='logs'):
    """
    Compare different models based on saved metrics
    
    Args:
        log_dir (str): Directory containing evaluation logs
        
    Returns:
        list: List of model metrics for comparison
    """
    # Find all metrics files in all subdirectories
    all_metrics = []
    for root, dirs, files in os.walk(log_dir):
        for file in files:
            if file.endswith('_metrics.json'):
                try:
                    metrics_path = os.path.join(root, file)
                    with open(metrics_path, 'r') as f:
                        metrics = json.load(f)
                        
                    # Try to get hyperparameters from summary file
                    summary_path = os.path.join(root, file.replace('_metrics.json', '_summary.json'))
                    if os.path.exists(summary_path):
                        with open(summary_path, 'r') as f:
                            summary = json.load(f)
                            if 'configuration' in summary:
                                metrics['hyperparameters'] = summary['configuration']
                            if 'preprocess_method' in summary:
                                metrics['preprocess_method'] = summary['preprocess_method']
                                
                    # Get directory name (date_sequence)
                    dir_name = os.path.basename(os.path.dirname(metrics_path))
                    metrics['dir_name'] = dir_name
                                
                    all_metrics.append(metrics)
                except Exception as e:
                    print(f"Error loading metrics from {metrics_path}: {str(e)}")
    
    if not all_metrics:
        print("No metrics files found")
        return []
    
    # Sort by accuracy
    all_metrics.sort(key=lambda x: x.get('accuracy', 0), reverse=True)
    
    # Print comparison
    print("\nModel Comparison:")
    headers = ['Run', 'Accuracy', 'F1 Score', 'Preprocess', 'Activation', 'Optimizer', 'BatchNorm', 'Dropout', 'LR', 'Batch', 'Weight Decay', 'Hidden Sizes']
    row_format = "{:<12} {:<10} {:<10} {:<11} {:<11} {:<10} {:<10} {:<10} {:<10} {:<8} {:<12} {:<20}"
    
    print(row_format.format(*headers))
    print("-" * 145)
    
    for m in all_metrics:
        # Get hyperparameters
        hp = m.get('hyperparameters', {})
        features = set(m.get('features', []))
        
        # Get preprocessing method
        preprocess = m.get('preprocess_method', 'standard').title()
        
        # Determine settings
        is_gelu = 'GELU' in features
        is_adam = 'Adam' in features
        has_bn = 'BatchNorm' in features or hp.get('use_bn', False) is True
        
        hidden_sizes = str(hp.get('hidden_sizes', 'N/A'))
        if len(hidden_sizes) > 17:
            hidden_sizes = hidden_sizes[:14] + '...'
            
        print(row_format.format(
            m.get('dir_name', 'Unknown'),
            f"{m.get('accuracy', 0):.4f}",
            f"{m.get('f1_score', 0):.4f}",
            preprocess,
            'GELU' if is_gelu else 'ReLU',
            'Adam' if is_adam else 'SGD',
            'Yes' if has_bn else 'No',
            str(hp.get('dropout', 'N/A')),
            str(hp.get('lr', 'N/A')),
            str(hp.get('batch_size', 'N/A')),
            str(hp.get('weight_decay', 'N/A')),
            hidden_sizes
        ))
    
    print("\nDetailed metrics saved in respective log directories")
    return all_metrics

if __name__ == '__main__':
    import sys
    log_dir = sys.argv[1] if len(sys.argv) > 1 else 'logs'
    compare_models(log_dir)