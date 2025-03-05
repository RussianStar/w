# LSTM-based Thermal Control System

This project implements a Long Short-Term Memory (LSTM) neural network for reinforcement learning to control a thermal system using the ThermalGymEnv environment. The implementation is designed to utilize NVIDIA GPUs for accelerated training, with support for multi-GPU setups.

## Project Overview

The system uses a Proximal Policy Optimization (PPO) algorithm with an LSTM-based policy network to learn optimal control strategies for maintaining temperatures in a thermal system while minimizing energy usage. The LSTM architecture allows the agent to capture temporal dependencies in the system dynamics, which is crucial for effective thermal control.

## Features

- LSTM-based policy network for temporal reasoning
- Multi-GPU support for accelerated training
- Comprehensive logging and visualization via TensorBoard
- Checkpoint saving and model evaluation during training
- Progress tracking with detailed metrics
- GPU monitoring and statistics logging
- Configurable hyperparameters via command-line arguments

## Requirements

- Python 3.7+
- PyTorch 1.13.0+
- Gymnasium 0.26.0+
- NumPy 1.18.0+
- TensorBoard (for logging)
- Matplotlib (for visualization)
- CUDA-compatible NVIDIA GPU(s) (for accelerated training)

## Project Structure

### Core Files

- `environment.py`: ThermalGymEnv implementation
- `simulation.py`: Thermal system simulation
- `config.json`: Configuration parameters for the thermal system

### LSTM Implementation

- `lstm_thermal_agent.py`: Core implementation of the LSTM-based PPO agent
- `train_lstm_thermal.py`: Training script with GPU optimization

### Utility Scripts

- `inference_demo.py`: Script for running inference with a trained model
- `monitor_gpu.py`: Tool for monitoring GPU usage during training
- `run_lstm_training.sh`: Shell script for running the training on a GPU workstation
- `run_training_with_monitoring.sh`: Script to run both training and GPU monitoring in parallel

### Documentation

- `LSTM_TRAINING_README.md`: Detailed documentation for the LSTM training process
- `LSTM_THERMAL_CONTROL_README.md`: This file, providing an overview of the entire project

## Quick Start

### Training with GPU Monitoring

To train the LSTM agent with GPU monitoring:

```bash
# Make the script executable
chmod +x run_training_with_monitoring.sh

# Run with default parameters
./run_training_with_monitoring.sh

# Or with custom parameters
./run_training_with_monitoring.sh --timesteps 5000000 --hidden_dim 512 --batch_size 256
```

### Running Inference

To run inference with a trained model:

```bash
python inference_demo.py --model_path models/lstm_ppo_thermal_best.pt
```

### Monitoring GPU Usage

To monitor GPU usage separately:

```bash
python monitor_gpu.py --interval 5 --log-file logs/gpu_stats.csv
```

## Detailed Usage

### Training Parameters

The training script supports various parameters to customize the training process:

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

### GPU Monitoring Parameters

The GPU monitoring script supports the following parameters:

- `--interval`: Interval between measurements in seconds (default: 5)
- `--log-file`: Path to CSV file to log statistics (default: None)
- `--duration`: Duration to monitor in seconds (default: indefinite)

### Inference Parameters

The inference script supports the following parameters:

- `--model_path`: Path to the trained model (default: "models/lstm_ppo_thermal_best.pt")
- `--config_file`: Path to the configuration file (default: "config.json")
- `--n_episodes`: Number of episodes to run (default: 3)
- `--max_steps`: Maximum number of steps per episode (default: 200)
- `--no_render`: Disable environment rendering
- `--no_plots`: Disable plot saving
- `--plot_dir`: Directory to save plots (default: "plots")

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
- `logs/gpu_stats_[timestamp].csv`: GPU statistics log (if enabled)
- `plots/lstm_inference_episode_[n].png`: Inference visualization plots

## Performance Considerations

- **Memory Usage**: LSTM-based policies require more memory than standard MLPs. If you encounter out-of-memory errors, try reducing `--hidden_dim` or `--mini_batch_size`.
- **Training Speed**: Multi-GPU training significantly accelerates the process but requires more memory. Monitor GPU utilization to ensure efficient resource usage.
- **Convergence**: LSTM-based policies may take longer to converge than simpler architectures. Be patient and monitor the evaluation rewards.

## Customization

The LSTM architecture and training parameters can be customized to suit different thermal control scenarios:

- For more complex thermal systems, increase `--hidden_dim` and `--lstm_layers`
- For faster convergence, adjust `--learning_rate` and `--mini_batch_size`
- For better exploration, modify the entropy coefficient in the agent

## Troubleshooting

- **GPU Out of Memory**: Reduce batch size or hidden dimension
- **Slow Training**: Check GPU utilization with `monitor_gpu.py`
- **Poor Performance**: Try increasing the number of training timesteps
- **Convergence Issues**: Adjust learning rate or try different hyperparameters

## License

This project is provided as-is for educational and research purposes.

## Acknowledgments

- The ThermalGymEnv environment is based on the Gymnasium framework
- The LSTM implementation uses PyTorch
- The PPO algorithm is inspired by the paper "Proximal Policy Optimization Algorithms" by Schulman et al.
