#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Training script for LSTM-based Reinforcement Learning on ThermalGymEnv

This script runs the LSTM-based PPO agent training on the ThermalGymEnv environment
with GPU acceleration. It's configured to utilize multiple GPUs if available and
provides detailed progress tracking during training.

Usage:
    python train_lstm_thermal.py [--args]
"""

import os
import argparse
import torch
from lstm_thermal_agent import main as lstm_main

if __name__ == "__main__":
    # Check CUDA availability and print GPU information
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"Found {device_count} CUDA device(s):")
        for i in range(device_count):
            print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
            print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
            print(f"    CUDA Capability: {torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}")
        
        if device_count > 1:
            print("\nMultiple GPUs detected - training will use all available GPUs")
    else:
        print("CUDA not available. This script is designed to run on GPU hardware.")
        print("Training will proceed on CPU, but will be significantly slower.")
    
    # Create parser
    parser = argparse.ArgumentParser(description="Train LSTM-based PPO agent for thermal control")
    
    # Add arguments with optimized defaults for GPU training
    parser.add_argument("--config_file", type=str, default="config.json", 
                        help="Path to the configuration file")
    
    # Model architecture
    parser.add_argument("--hidden_dim", type=int, default=256, 
                        help="Hidden dimension of the LSTM (increased for better performance)")
    parser.add_argument("--lstm_layers", type=int, default=2, 
                        help="Number of LSTM layers")
    
    # Training hyperparameters
    parser.add_argument("--learning_rate", type=float, default=3e-4, 
                        help="Learning rate")
    parser.add_argument("--total_timesteps", type=int, default=2000000, 
                        help="Total number of timesteps to train for")
    parser.add_argument("--rollout_length", type=int, default=2048, 
                        help="Number of timesteps to collect before updating")
    parser.add_argument("--mini_batch_size", type=int, default=256, 
                        help="Mini-batch size (increased for GPU efficiency)")
    
    # GPU utilization
    parser.add_argument("--multi_gpu", action="store_true", default=True,
                        help="Whether to use multiple GPUs (default: True if multiple GPUs available)")
    
    # Logging and saving
    parser.add_argument("--save_freq", type=int, default=50000, 
                        help="Frequency of saving the model")
    parser.add_argument("--log_freq", type=int, default=1000, 
                        help="Frequency of logging metrics")
    parser.add_argument("--eval_freq", type=int, default=50000, 
                        help="Frequency of evaluating the agent")
    parser.add_argument("--save_path", type=str, default="models/lstm_ppo_thermal", 
                        help="Path to save the model")
    parser.add_argument("--log_path", type=str, default="logs/lstm_ppo_thermal", 
                        help="Path to save the logs")
    
    # Parse arguments
    args = parser.parse_args()
    
    # If multiple GPUs are available but multi_gpu wasn't explicitly set to False,
    # ensure it's set to True
    if torch.cuda.device_count() > 1 and '--multi_gpu' not in os.sys.argv:
        args.multi_gpu = True
        print("Automatically enabling multi-GPU training")
    
    # Print training configuration
    print("\n" + "="*50)
    print("LSTM THERMAL CONTROL TRAINING CONFIGURATION")
    print("="*50)
    print(f"Training Steps: {args.total_timesteps}")
    print(f"LSTM Architecture: {args.hidden_dim} units, {args.lstm_layers} layers")
    print(f"Batch Size: {args.mini_batch_size}")
    print(f"Multi-GPU: {'Enabled' if args.multi_gpu else 'Disabled'}")
    print(f"Model Save Path: {args.save_path}")
    print("="*50 + "\n")
    
    # Create directories
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    os.makedirs(args.log_path, exist_ok=True)
    
    # Run the main function from lstm_thermal_agent.py
    print("Starting training...")
    lstm_main(args)
    print("\nTraining completed!")
    print(f"Trained models saved to: {args.save_path}_[timestep/best/final].pt")
    print(f"Training logs saved to: {args.log_path}")
