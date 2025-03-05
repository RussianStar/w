# LSTM-based Reinforcement Learning for Thermal System Control

This project implements a Long Short-Term Memory (LSTM) neural network for reinforcement learning to control a thermal system using the ThermalGymEnv environment. The implementation is designed to utilize NVIDIA GPUs for accelerated training.

## Overview

The system uses a Proximal Policy Optimization (PPO) algorithm with an LSTM-based policy network to learn optimal control strategies for maintaining temperatures in a thermal system while minimizing energy usage. The LSTM architecture allows the agent to capture temporal dependencies in the system dynamics, which is crucial for effective thermal control.

## Features

- LSTM-based policy network for temporal reasoning
- Multi-GPU support for accelerated training
- Comprehensive logging and visualization via TensorBoard
- Checkpoint saving and model evaluation during training
- Progress tracking with detailed metrics
- Configurable hyperparameters via command-line arguments

## Requirements

- Python 3.7+
- PyTorch 1.13.0+
- Gymnasium 0.26.0+
- NumPy 1.18.0+
- TensorBoard (for logging)
- CUDA-compatible NVIDIA GPU(s) (for accelerated training)

## Files

- `lstm_thermal_agent.py`: Core implementation of the LSTM-based PPO agent
- `train_lstm_thermal.py`: Training script with GPU optimization
- `environment.py`: ThermalGymEnv implementation
- `simulation.py`: Thermal system simulation
- `config.json`: Configuration parameters for the thermal system

## Usage

### Training

To train the LSTM agent with default parameters:

```bash
python train_lstm_thermal.py
```

### Training with Custom Parameters

```bash
python train_lstm_thermal.py --hidden_dim 512 --lstm_layers 3 --total_timesteps 5000000
```

### Common Parameters

- `--hidden_dim`: Hidden dimension of the LSTM (default: 256)
- `--lstm_layers`: Number of LSTM layers (default: 2)
- `--learning_rate`: Learning rate (default: 3e-4)
- `--total_timesteps`: Total number of timesteps to train for (default: 2,000,000)
- `--mini_batch_size`: Mini-batch size (default: 256)
- `--multi_gpu`: Whether to use multiple GPUs (default: True if multiple GPUs available)
- `--save_path`: Path to save the model (default: "models/lstm_ppo_thermal")
- `--log_path`: Path to save the logs (default: "logs/lstm_ppo_thermal")

For a complete list of parameters, run:

```bash
python train_lstm_thermal.py --help
```

## GPU Utilization

The training script automatically detects available GPUs and configures the training process to utilize them efficiently:

1. If multiple GPUs are available, the script enables multi-GPU training by default
2. The LSTM policy network is distributed across available GPUs using PyTorch's DataParallel
3. Batch sizes are automatically optimized for GPU training

## Monitoring Training

Training progress is logged to TensorBoard. To view the logs:

```bash
tensorboard --logdir=logs/lstm_ppo_thermal
```

This will display metrics including:
- Episode rewards
- Policy, value, and entropy losses
- Training speed (steps per second)
- Evaluation performance

## Output Files

During training, the following files are generated:

- `models/lstm_ppo_thermal_[timestep].pt`: Periodic model checkpoints
- `models/lstm_ppo_thermal_best.pt`: Best model based on evaluation performance
- `models/lstm_ppo_thermal_final.pt`: Final model after training completion
- `logs/lstm_ppo_thermal/`: TensorBoard logs

## Performance Considerations

- **Memory Usage**: LSTM-based policies require more memory than standard MLPs. If you encounter out-of-memory errors, try reducing `--hidden_dim` or `--mini_batch_size`.
- **Training Speed**: Multi-GPU training significantly accelerates the process but requires more memory. Monitor GPU utilization to ensure efficient resource usage.
- **Convergence**: LSTM-based policies may take longer to converge than simpler architectures. Be patient and monitor the evaluation rewards.

## Evaluation

To evaluate a trained model without further training:

```bash
python lstm_thermal_agent.py --eval_only --save_path models/lstm_ppo_thermal_best
```

This will load the specified model and run evaluation episodes to assess its performance.
