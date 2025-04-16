#!/bin/bash

# Create log directory
mkdir -p logs/ablation_new

# Define common parameters
COMMON_PARAMS="--dropout 0.5 --lr 0.0005 --batch_size 128 --weight_decay 0.0001 --hidden_sizes 512 256 --epochs 100 --early_stopping_patience 10 --log_dir logs/ablation_new"

echo "Starting Ablation Study Experiments (24 configurations)..."
echo "All results will be saved in logs/ablation_new directory"
echo "========================================"

# Group 1: Standard preprocessing + GELU + Adam + Batch Norm (optimal model)
echo "Running Experiment 1/24: Standard + GELU + Adam + Batch Norm"
python3 main.py --gelu --adam ${COMMON_PARAMS} --preprocess standard

# Group 2: Standard preprocessing + GELU + Adam + No Batch Norm
echo "Running Experiment 2/24: Standard + GELU + Adam + No Batch Norm"
python3 main.py --gelu --adam ${COMMON_PARAMS} --preprocess standard --no_batch_norm

# Group 3: Standard preprocessing + GELU + SGD + Batch Norm
echo "Running Experiment 3/24: Standard + GELU + SGD + Batch Norm"
python3 main.py --gelu ${COMMON_PARAMS} --preprocess standard

# Group 4: Standard preprocessing + GELU + SGD + No Batch Norm
echo "Running Experiment 4/24: Standard + GELU + SGD + No Batch Norm"
python3 main.py --gelu ${COMMON_PARAMS} --preprocess standard --no_batch_norm

# Group 5: Standard preprocessing + ReLU + Adam + Batch Norm
echo "Running Experiment 5/24: Standard + ReLU + Adam + Batch Norm"
python3 main.py --adam ${COMMON_PARAMS} --preprocess standard

# Group 6: Standard preprocessing + ReLU + Adam + No Batch Norm
echo "Running Experiment 6/24: Standard + ReLU + Adam + No Batch Norm"
python3 main.py --adam ${COMMON_PARAMS} --preprocess standard --no_batch_norm

# Group 7: Standard preprocessing + ReLU + SGD + Batch Norm
echo "Running Experiment 7/24: Standard + ReLU + SGD + Batch Norm"
python3 main.py ${COMMON_PARAMS} --preprocess standard

# Group 8: Standard preprocessing + ReLU + SGD + No Batch Norm
echo "Running Experiment 8/24: Standard + ReLU + SGD + No Batch Norm"
python3 main.py ${COMMON_PARAMS} --preprocess standard --no_batch_norm

# Group 9: No preprocessing + GELU + Adam + Batch Norm
echo "Running Experiment 9/24: None + GELU + Adam + Batch Norm"
python3 main.py --gelu --adam ${COMMON_PARAMS} --preprocess none

# Group 10: No preprocessing + GELU + Adam + No Batch Norm
echo "Running Experiment 10/24: None + GELU + Adam + No Batch Norm"
python3 main.py --gelu --adam ${COMMON_PARAMS} --preprocess none --no_batch_norm

# Group 11: No preprocessing + GELU + SGD + Batch Norm
echo "Running Experiment 11/24: None + GELU + SGD + Batch Norm"
python3 main.py --gelu ${COMMON_PARAMS} --preprocess none

# Group 12: No preprocessing + GELU + SGD + No Batch Norm
echo "Running Experiment 12/24: None + GELU + SGD + No Batch Norm"
python3 main.py --gelu ${COMMON_PARAMS} --preprocess none --no_batch_norm

# Group 13: No preprocessing + ReLU + Adam + Batch Norm
echo "Running Experiment 13/24: None + ReLU + Adam + Batch Norm"
python3 main.py --adam ${COMMON_PARAMS} --preprocess none

# Group 14: No preprocessing + ReLU + Adam + No Batch Norm
echo "Running Experiment 14/24: None + ReLU + Adam + No Batch Norm"
python3 main.py --adam ${COMMON_PARAMS} --preprocess none --no_batch_norm

# Group 15: No preprocessing + ReLU + SGD + Batch Norm
echo "Running Experiment 15/24: None + ReLU + SGD + Batch Norm"
python3 main.py ${COMMON_PARAMS} --preprocess none

# Group 16: No preprocessing + ReLU + SGD + No Batch Norm
echo "Running Experiment 16/24: None + ReLU + SGD + No Batch Norm"
python3 main.py ${COMMON_PARAMS} --preprocess none --no_batch_norm

# Group 17: MinMax preprocessing + GELU + Adam + Batch Norm
echo "Running Experiment 17/24: MinMax + GELU + Adam + Batch Norm"
python3 main.py --gelu --adam ${COMMON_PARAMS} --preprocess minmax

# Group 18: MinMax preprocessing + GELU + Adam + No Batch Norm
echo "Running Experiment 18/24: MinMax + GELU + Adam + No Batch Norm"
python3 main.py --gelu --adam ${COMMON_PARAMS} --preprocess minmax --no_batch_norm

# Group 19: MinMax preprocessing + GELU + SGD + Batch Norm
echo "Running Experiment 19/24: MinMax + GELU + SGD + Batch Norm"
python3 main.py --gelu ${COMMON_PARAMS} --preprocess minmax

# Group 20: MinMax preprocessing + GELU + SGD + No Batch Norm
echo "Running Experiment 20/24: MinMax + GELU + SGD + No Batch Norm"
python3 main.py --gelu ${COMMON_PARAMS} --preprocess minmax --no_batch_norm

# Group 21: MinMax preprocessing + ReLU + Adam + Batch Norm
echo "Running Experiment 21/24: MinMax + ReLU + Adam + Batch Norm"
python3 main.py --adam ${COMMON_PARAMS} --preprocess minmax

# Group 22: MinMax preprocessing + ReLU + Adam + No Batch Norm
echo "Running Experiment 22/24: MinMax + ReLU + Adam + No Batch Norm"
python3 main.py --adam ${COMMON_PARAMS} --preprocess minmax --no_batch_norm

# Group 23: MinMax preprocessing + ReLU + SGD + Batch Norm
echo "Running Experiment 23/24: MinMax + ReLU + SGD + Batch Norm"
python3 main.py ${COMMON_PARAMS} --preprocess minmax

# Group 24: MinMax preprocessing + ReLU + SGD + No Batch Norm
echo "Running Experiment 24/24: MinMax + ReLU + SGD + No Batch Norm"
python3 main.py ${COMMON_PARAMS} --preprocess minmax --no_batch_norm

echo "========================================"
echo "All experiments completed!"
echo "Use the following command to compare model performance:"
echo "python3 -c \"from evaluate import ModelEvaluator; ModelEvaluator.compare_models('logs/ablation_new')\""