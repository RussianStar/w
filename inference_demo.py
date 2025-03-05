#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Inference Demo for LSTM-based Thermal Control

This script demonstrates how to use a trained LSTM-based PPO model
for inference on the ThermalGymEnv environment. It loads a trained model
and runs it on the environment, visualizing the results.

Usage:
    python inference_demo.py [--model_path models/lstm_ppo_thermal_best.pt]
"""

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from environment import ThermalGymEnv
from lstm_thermal_agent import LSTMActorCritic, LSTMPPOAgent

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def run_inference(
    model_path,
    config_file="config.json",
    n_episodes=3,
    max_steps=200,
    render=True,
    save_plots=True,
    plot_dir="plots"
):
    """
    Run inference using a trained LSTM-PPO model.
    
    Args:
        model_path (str): Path to the trained model
        config_file (str): Path to the configuration file
        n_episodes (int): Number of episodes to run
        max_steps (int): Maximum number of steps per episode
        render (bool): Whether to render the environment
        save_plots (bool): Whether to save plots
        plot_dir (str): Directory to save plots
    """
    # Create environment
    env = ThermalGymEnv(config_file)
    
    # Get state and action dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    
    # Create agent
    agent = LSTMPPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=256,  # Match the training hidden dimension
        lstm_layers=2    # Match the training LSTM layers
    )
    
    # Load model
    agent.load(model_path)
    print(f"Model loaded from {model_path}")
    
    # Create plot directory if needed
    if save_plots:
        os.makedirs(plot_dir, exist_ok=True)
    
    # Run episodes
    for episode in range(n_episodes):
        print(f"\nRunning episode {episode + 1}/{n_episodes}")
        
        # Reset environment and agent hidden state
        state, _ = env.reset()
        agent.policy.reset_hidden_state()
        
        # Initialize episode variables
        episode_reward = 0
        done = False
        step = 0
        
        # Initialize data collection for plotting
        temperatures = []
        end_sensor_temps = []
        outside_temps = []
        opening_degrees = []
        rewards = []
        steps = []
        
        # Episode loop
        while not done and step < max_steps:
            # Get action
            action = agent.get_action(state, deterministic=True)
            
            # Take step in environment
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Render if requested
            if render:
                env.render()
            
            # Update state and reward
            state = next_state
            episode_reward += reward
            
            # Collect data for plotting
            temperatures.append(info["temperatures"].copy())
            end_sensor_temps.append(info["end_sensor_temp"])
            outside_temps.append(info["outside_temp"])
            opening_degrees.append(env.sim.opening_degrees.copy())
            rewards.append(reward)
            steps.append(step)
            
            # Increment step
            step += 1
            
            # Print step info every 10 steps
            if step % 10 == 0:
                print(f"Step {step}: Reward = {reward:.2f}, "
                      f"End Sensor Temp = {info['end_sensor_temp']:.2f}, "
                      f"Constraints Met = {info['constraints_met']}")
        
        # Print episode summary
        print(f"Episode {episode + 1} completed:")
        print(f"  Steps: {step}")
        print(f"  Total Reward: {episode_reward:.2f}")
        print(f"  Final End Sensor Temperature: {end_sensor_temps[-1]:.2f}")
        print(f"  Minimum Legal Temperature: {env.sim.T_law:.2f}")
        
        # Create plots
        if save_plots:
            # Convert data to numpy arrays for easier plotting
            temperatures = np.array(temperatures)
            opening_degrees = np.array(opening_degrees)
            
            # Create figure with subplots
            fig, axs = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
            
            # Plot temperatures
            for i in range(temperatures.shape[1]):
                axs[0].plot(steps, temperatures[:, i], label=f"Pipe {i+1}")
            axs[0].plot(steps, end_sensor_temps, label="End Sensor", linestyle="--", linewidth=2)
            axs[0].axhline(y=env.sim.T_law, color="r", linestyle="-.", label="Legal Minimum")
            axs[0].set_ylabel("Temperature (°C)")
            axs[0].set_title("Pipe Temperatures")
            axs[0].legend()
            axs[0].grid(True)
            
            # Plot opening degrees
            for i in range(opening_degrees.shape[1]):
                axs[1].plot(steps, opening_degrees[:, i], label=f"Pipe {i+1}")
            axs[1].set_ylabel("Opening Degree")
            axs[1].set_title("Valve Opening Degrees")
            axs[1].legend()
            axs[1].grid(True)
            
            # Plot rewards
            axs[2].plot(steps, rewards)
            axs[2].set_xlabel("Step")
            axs[2].set_ylabel("Reward")
            axs[2].set_title("Rewards")
            axs[2].grid(True)
            
            # Adjust layout and save
            plt.tight_layout()
            plt.savefig(f"{plot_dir}/lstm_inference_episode_{episode+1}.png")
            plt.close()
            
            print(f"Plot saved to {plot_dir}/lstm_inference_episode_{episode+1}.png")
    
    print("\nInference completed!")


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Run inference with trained LSTM-PPO model")
    parser.add_argument("--model_path", type=str, default="models/lstm_ppo_thermal_best.pt",
                        help="Path to the trained model")
    parser.add_argument("--config_file", type=str, default="config.json",
                        help="Path to the configuration file")
    parser.add_argument("--n_episodes", type=int, default=3,
                        help="Number of episodes to run")
    parser.add_argument("--max_steps", type=int, default=200,
                        help="Maximum number of steps per episode")
    parser.add_argument("--no_render", action="store_true",
                        help="Disable environment rendering")
    parser.add_argument("--no_plots", action="store_true",
                        help="Disable plot saving")
    parser.add_argument("--plot_dir", type=str, default="plots",
                        help="Directory to save plots")
    
    args = parser.parse_args()
    
    # Run inference
    run_inference(
        model_path=args.model_path,
        config_file=args.config_file,
        n_episodes=args.n_episodes,
        max_steps=args.max_steps,
        render=not args.no_render,
        save_plots=not args.no_plots,
        plot_dir=args.plot_dir
    )
