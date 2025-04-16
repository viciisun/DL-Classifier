#!/bin/bash

# Create log directory
mkdir -p logs/hyperparams_new

# Base parameters (optimal model without batch norm)
BASE_PARAMS="--gelu --adam --no_batch_norm --preprocess standard --epochs 100 --early_stopping_patience 10 --batch_size 128 --log_dir logs/hyperparams_new"

# Hyperparameter values to explore
LEARNING_RATES=("0.0001" "0.0005" "0.01")
DROPOUTS=("0.5" "0.3" "0.7")
HIDDEN_LAYERS=("512 256" "256 128")
WEIGHT_DECAYS=("0.0001")

# Calculate total combinations
TOTAL=$((${#LEARNING_RATES[@]} * ${#DROPOUTS[@]} * ${#HIDDEN_LAYERS[@]} * ${#WEIGHT_DECAYS[@]}))
echo "Starting Hyperparameter Analysis with $TOTAL configurations..."
echo "All results will be saved in logs/hyperparams directory"
echo "========================================"

# Counter for experiments
COUNT=0

# Loop through all hyperparameter combinations
for lr in "${LEARNING_RATES[@]}"; do
    for dropout in "${DROPOUTS[@]}"; do
        for hidden in "${HIDDEN_LAYERS[@]}"; do
            for wd in "${WEIGHT_DECAYS[@]}"; do
                # Increment counter
                COUNT=$((COUNT + 1))
                
                # Define experiment name for logging
                EXP_NAME="lr${lr}_drop${dropout}_wd${wd}_h${hidden// /_}"
                
                echo "Running Experiment $COUNT/$TOTAL: LR=$lr, Dropout=$dropout, Weight Decay=$wd, Hidden=[${hidden}]"
                python3 main.py ${BASE_PARAMS} --lr ${lr} --dropout ${dropout} --weight_decay ${wd} --hidden_sizes ${hidden} --model_name ${EXP_NAME}
                
                echo "Completed $COUNT/$TOTAL experiments"
                echo "--------------------"
            done
        done
    done
done

echo "========================================"
echo "All experiments completed!"
echo "Total configurations tested: $TOTAL"
echo "Use the following command to compare model performance:"
echo "python3 -c \"from evaluate import ModelEvaluator; ModelEvaluator.compare_models('logs/hyperparams')\""