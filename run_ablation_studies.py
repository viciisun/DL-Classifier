#!/usr/bin/env python3

import subprocess
import os
import time
from datetime import datetime

def run_command(command):
    """Run command and print output"""
    print(f"Running: {command}")
    
    start_time = time.time()
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    while True:
        output = process.stdout.readline().decode('utf-8')
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())
    
    # Process any remaining output
    stdout, stderr = process.communicate()
    if stdout:
        print(stdout.decode('utf-8').strip())
    if stderr:
        print(stderr.decode('utf-8').strip())
    
    end_time = time.time()
    print(f"Command completed in {end_time - start_time:.2f} seconds")
    print("-" * 80)
    
    return process.returncode

def run_ablation_studies():
    """Run all ablation studies"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"logs/ablation_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"Running ablation studies, logs will be saved to {log_dir}")
    
    # Define all experiments to run
    experiments = [
        # Baseline model (default features)
        f"python3 main.py --log_dir {log_dir}",
        
        # Individual features
        f"python3 main.py GELU --log_dir {log_dir}",
        f"python3 main.py Adam --log_dir {log_dir}",
        f"python3 main.py preprocess --log_dir {log_dir}",
        
        # Preprocessing methods
        f"python3 main.py preprocess --preprocess_method standard --log_dir {log_dir}",
        f"python3 main.py preprocess --preprocess_method minmax --log_dir {log_dir}",
        f"python3 main.py preprocess --preprocess_method pca --pca_components 64 --log_dir {log_dir}",
        
        # Combined features
        f"python3 main.py GELU Adam --log_dir {log_dir}",
        f"python3 main.py GELU preprocess --log_dir {log_dir}",
        f"python3 main.py Adam preprocess --log_dir {log_dir}",
        
        # Different architectures
        f"python3 main.py --hidden_sizes 512 256 --log_dir {log_dir}",
        f"python3 main.py --hidden_sizes 512 256 128 64 --log_dir {log_dir}",
        
        # Different learning rates
        f"python3 main.py --lr 0.01 --log_dir {log_dir}",
        f"python3 main.py --lr 0.0001 --log_dir {log_dir}",
        
        # Full model with all features
        f"python3 main.py full --log_dir {log_dir}"
    ]
    
    # Run all experiments
    for i, experiment in enumerate(experiments):
        print(f"Experiment {i+1}/{len(experiments)}")
        run_command(experiment)
    
    print("\nAll ablation studies completed!")
    print(f"Results saved to {log_dir}")
    
    # Provide instructions for analyzing results
    print("\nTo compare all models, run:")
    print(f"python3 -c \"from evaluate import ModelEvaluator; ModelEvaluator.compare_models('{log_dir}')\"")

if __name__ == "__main__":
    run_ablation_studies()