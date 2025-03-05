#!/bin/bash
# Run LSTM-based Thermal Control Training on GPU Workstation
# This script sets up the environment and launches the training with appropriate parameters

# Exit on error
set -e

# Print header
echo "============================================================"
echo "LSTM-based Thermal Control Training on GPU Workstation"
echo "============================================================"

# Check for NVIDIA GPUs
if command -v nvidia-smi &> /dev/null; then
    echo "Checking GPU configuration..."
    nvidia-smi
    
    # Get number of GPUs
    NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    echo "Detected $NUM_GPUS NVIDIA GPU(s)"
    
    if [ $NUM_GPUS -lt 1 ]; then
        echo "Error: No NVIDIA GPUs detected. This script requires at least one NVIDIA GPU."
        exit 1
    fi
    
    if [ $NUM_GPUS -gt 1 ]; then
        echo "Multi-GPU training will be enabled automatically"
    fi
else
    echo "Error: nvidia-smi not found. This script requires NVIDIA GPUs and drivers."
    exit 1
fi

# Create directories
mkdir -p models logs plots

# Parse command line arguments
TIMESTEPS=2000000
HIDDEN_DIM=256
BATCH_SIZE=256
EVAL_EPISODES=5

# Process command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --timesteps)
        TIMESTEPS="$2"
        shift
        shift
        ;;
        --hidden_dim)
        HIDDEN_DIM="$2"
        shift
        shift
        ;;
        --batch_size)
        BATCH_SIZE="$2"
        shift
        shift
        ;;
        --eval_episodes)
        EVAL_EPISODES="$2"
        shift
        shift
        ;;
        *)
        echo "Unknown option: $1"
        echo "Usage: $0 [--timesteps N] [--hidden_dim N] [--batch_size N] [--eval_episodes N]"
        exit 1
        ;;
    esac
done

echo "Configuration:"
echo "  Total Timesteps: $TIMESTEPS"
echo "  Hidden Dimension: $HIDDEN_DIM"
echo "  Batch Size: $BATCH_SIZE"
echo "  Evaluation Episodes: $EVAL_EPISODES"
echo "============================================================"

# Set environment variables for better GPU performance
export CUDA_VISIBLE_DEVICES=0,1  # Use first two GPUs if available
export TF_FORCE_GPU_ALLOW_GROWTH=true  # For TensorBoard

# Run the training script
echo "Starting training..."
python train_lstm_thermal.py \
    --total_timesteps $TIMESTEPS \
    --hidden_dim $HIDDEN_DIM \
    --mini_batch_size $BATCH_SIZE \
    --eval_episodes $EVAL_EPISODES

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo "============================================================"
    echo "Training completed successfully!"
    echo "Model saved to: models/lstm_ppo_thermal_*.pt"
    echo "Logs saved to: logs/lstm_ppo_thermal/"
    echo ""
    echo "To view training progress with TensorBoard:"
    echo "  tensorboard --logdir=logs/lstm_ppo_thermal"
    echo ""
    echo "To run inference with the trained model:"
    echo "  python inference_demo.py --model_path models/lstm_ppo_thermal_best.pt"
    echo "============================================================"
else
    echo "Training failed with error code $?"
    exit 1
fi
