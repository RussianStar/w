#!/bin/bash
# Run LSTM Training with GPU Monitoring
# This script launches both the LSTM training and GPU monitoring in parallel

# Exit on error in the main script
set -e

# Print header
echo "============================================================"
echo "LSTM Training with GPU Monitoring"
echo "============================================================"

# Check for NVIDIA GPUs
if ! command -v nvidia-smi &> /dev/null; then
    echo "Error: nvidia-smi not found. This script requires NVIDIA GPUs and drivers."
    exit 1
fi

# Create directories
mkdir -p models logs plots

# Parse command line arguments
TIMESTEPS=2000000
HIDDEN_DIM=256
BATCH_SIZE=256
MONITOR_INTERVAL=5
LOG_GPU_STATS=true

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
        --monitor_interval)
        MONITOR_INTERVAL="$2"
        shift
        shift
        ;;
        --no_gpu_logging)
        LOG_GPU_STATS=false
        shift
        ;;
        *)
        echo "Unknown option: $1"
        echo "Usage: $0 [--timesteps N] [--hidden_dim N] [--batch_size N] [--monitor_interval N] [--no_gpu_logging]"
        exit 1
        ;;
    esac
done

# Print configuration
echo "Configuration:"
echo "  Total Timesteps: $TIMESTEPS"
echo "  Hidden Dimension: $HIDDEN_DIM"
echo "  Batch Size: $BATCH_SIZE"
echo "  GPU Monitor Interval: $MONITOR_INTERVAL seconds"
echo "  Log GPU Statistics: $LOG_GPU_STATS"
echo "============================================================"

# Set environment variables for better GPU performance
export CUDA_VISIBLE_DEVICES=0,1  # Use first two GPUs if available
export TF_FORCE_GPU_ALLOW_GROWTH=true  # For TensorBoard

# Get current timestamp for log files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
GPU_LOG_FILE="logs/gpu_stats_${TIMESTAMP}.csv"

# Start GPU monitoring in the background if enabled
if [ "$LOG_GPU_STATS" = true ]; then
    echo "Starting GPU monitoring..."
    python monitor_gpu.py --interval $MONITOR_INTERVAL --log-file $GPU_LOG_FILE &
    MONITOR_PID=$!
    echo "GPU monitoring started with PID $MONITOR_PID"
    echo "GPU statistics will be logged to $GPU_LOG_FILE"
    echo "============================================================"
fi

# Run the training script
echo "Starting LSTM training..."
python train_lstm_thermal.py \
    --total_timesteps $TIMESTEPS \
    --hidden_dim $HIDDEN_DIM \
    --mini_batch_size $BATCH_SIZE

# Check if training completed successfully
TRAINING_STATUS=$?
if [ $TRAINING_STATUS -eq 0 ]; then
    echo "============================================================"
    echo "Training completed successfully!"
    echo "Model saved to: models/lstm_ppo_thermal_*.pt"
    echo "Logs saved to: logs/lstm_ppo_thermal/"
    
    if [ "$LOG_GPU_STATS" = true ]; then
        echo "GPU statistics logged to: $GPU_LOG_FILE"
    fi
    
    echo ""
    echo "To view training progress with TensorBoard:"
    echo "  tensorboard --logdir=logs/lstm_ppo_thermal"
    echo ""
    echo "To run inference with the trained model:"
    echo "  python inference_demo.py --model_path models/lstm_ppo_thermal_best.pt"
    echo "============================================================"
else
    echo "Training failed with error code $TRAINING_STATUS"
fi

# Stop GPU monitoring if it was started
if [ "$LOG_GPU_STATS" = true ] && [ -n "$MONITOR_PID" ]; then
    echo "Stopping GPU monitoring (PID $MONITOR_PID)..."
    kill $MONITOR_PID 2>/dev/null || true
fi

# Exit with the training status
exit $TRAINING_STATUS
