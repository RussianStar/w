"""
Optional module for implementing reinforcement learning models to control the thermal system.

This module provides examples of how to implement and train RL agents using the
ThermalGymEnv environment. It includes implementations using stable-baselines3.

Note: To use this module, you'll need to install additional dependencies:
    pip install stable-baselines3
"""

import numpy as np
import os
import time
from stable_baselines3 import PPO, SAC, A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from environment import ThermalGymEnv


def train_ppo_agent(config_file="config.json", total_timesteps=100000, save_path="models/ppo_thermal"):
    """
    Train a PPO agent on the thermal environment.
    
    Args:
        config_file (str): Path to the configuration file
        total_timesteps (int): Total number of training timesteps
        save_path (str): Path to save the trained model
        
    Returns:
        PPO: Trained PPO model
    """
    # Create and wrap the environment
    env = ThermalGymEnv(config_file)
    
    # Create log directory
    os.makedirs("logs", exist_ok=True)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Create the evaluation environment
    eval_env = ThermalGymEnv(config_file)
    
    # Create callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{os.path.dirname(save_path)}/best_model",
        log_path="logs/eval_results",
        eval_freq=5000,
        deterministic=True,
        render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=f"{os.path.dirname(save_path)}/checkpoints",
        name_prefix="ppo_thermal"
    )
    
    # Create the PPO agent
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=False,
        sde_sample_freq=-1,
        target_kl=None,
        tensorboard_log="logs/ppo_tensorboard/",
        verbose=1
    )
    
    # Train the agent
    print("Starting training...")
    start_time = time.time()
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback]
    )
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Save the final model
    model.save(save_path)
    print(f"Model saved to {save_path}")
    
    return model


def train_sac_agent(config_file="config.json", total_timesteps=100000, save_path="models/sac_thermal"):
    """
    Train a SAC agent on the thermal environment.
    
    Args:
        config_file (str): Path to the configuration file
        total_timesteps (int): Total number of training timesteps
        save_path (str): Path to save the trained model
        
    Returns:
        SAC: Trained SAC model
    """
    # Create and wrap the environment
    env = ThermalGymEnv(config_file)
    
    # Create log directory
    os.makedirs("logs", exist_ok=True)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Create the SAC agent
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=1000000,
        learning_starts=100,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        action_noise=None,
        replay_buffer_class=None,
        replay_buffer_kwargs=None,
        optimize_memory_usage=False,
        ent_coef="auto",
        target_update_interval=1,
        target_entropy="auto",
        use_sde=False,
        sde_sample_freq=-1,
        use_sde_at_warmup=False,
        tensorboard_log="logs/sac_tensorboard/",
        verbose=1
    )
    
    # Train the agent
    print("Starting training...")
    start_time = time.time()
    model.learn(total_timesteps=total_timesteps)
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Save the final model
    model.save(save_path)
    print(f"Model saved to {save_path}")
    
    return model


def evaluate_agent(model, config_file="config.json", n_eval_episodes=10, render=True):
    """
    Evaluate a trained agent on the thermal environment.
    
    Args:
        model: Trained RL model
        config_file (str): Path to the configuration file
        n_eval_episodes (int): Number of evaluation episodes
        render (bool): Whether to render the environment during evaluation
        
    Returns:
        tuple: (mean_reward, std_reward)
    """
    # Create the environment
    env = ThermalGymEnv(config_file)
    
    # Evaluate the agent
    mean_reward, std_reward = evaluate_policy(
        model,
        env,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
        return_episode_rewards=False
    )
    
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    # Run a demonstration if render is True
    if render:
        obs, _ = env.reset()
        terminated = False
        truncated = False
        total_reward = 0
        step = 0
        
        print("\nRunning demonstration episode...")
        while not (terminated or truncated) and step < 100:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            total_reward += reward
            step += 1
            
        print(f"Demonstration episode completed with reward: {total_reward:.2f}")
    
    return mean_reward, std_reward


def run_custom_policy(config_file="config.json", n_steps=100):
    """
    Run a custom hand-crafted policy on the thermal environment.
    This demonstrates how to implement a simple rule-based controller.
    
    Args:
        config_file (str): Path to the configuration file
        n_steps (int): Number of steps to run
        
    Returns:
        float: Total reward
    """
    # Create the environment
    env = ThermalGymEnv(config_file)
    
    # Reset the environment
    state, _ = env.reset()
    
    # Extract parameters
    n_pipes = env.sim.n_pipes
    T_law = env.sim.T_law
    
    # Run the custom policy
    total_reward = 0
    for step in range(n_steps):
        # Get current temperatures (all except the last element, which is outside temp)
        temperatures = state[:-1]
        
        # Simple rule-based policy:
        # - If temperature is close to T_law, open valve more
        # - If temperature is well above T_law, close valve to save energy
        action = np.zeros(n_pipes)
        for i in range(n_pipes):
            temp_margin = temperatures[i] - T_law
            
            if temp_margin < 5.0:
                # Temperature is getting close to minimum, open valve more
                action[i] = 0.8
            elif temp_margin < 10.0:
                # Moderate margin, keep valve partially open
                action[i] = 0.5
            else:
                # Large margin, save energy by closing valve more
                action[i] = 0.2
        
        # Apply the action
        state, reward, terminated, truncated, info = env.step(action)
        
        # Render the environment
        env.render()
        
        # Accumulate reward
        total_reward += reward
        
        # Check if episode is done
        if terminated or truncated:
            print(f"Episode ended after {step + 1} steps due to constraint violation!")
            break
    
    print(f"Custom policy completed with total reward: {total_reward:.2f}")
    return total_reward


if __name__ == "__main__":
    # Example usage
    print("=== Training PPO Agent ===")
    ppo_model = train_ppo_agent(total_timesteps=10000)  # Reduced for demonstration
    
    print("\n=== Evaluating PPO Agent ===")
    evaluate_agent(ppo_model, n_eval_episodes=3)
    
    print("\n=== Running Custom Policy ===")
    run_custom_policy(n_steps=50)

    print("\n=== Estimating Parameters with PPO ===")
    data = np.random.rand(100, 12)  # Example data: 100 timesteps, 12 features (4 pipes + end sensor + outside temp)
    estimated_params = estimate_parameters_with_ppo(data)
    print(f"Estimated parameters: {estimated_params}")

def estimate_parameters_with_ppo(timeseries_data):
    """
    Estimate the underlying parameters from time series sensor/valve data using a PPO model.
    
    Args:
        timeseries_data (numpy.ndarray): Time series data of shape (timesteps, features)
        
    Returns:
        numpy.ndarray: Estimated parameters
    """
    # Load the trained PPO model
    ppo_model = PPO.load("models/lstm_thermal")
    
    # Use the model to predict actions based on the timeseries_data
    estimated_actions, _states = ppo_model.predict(timeseries_data)
    
    # For demonstration purposes, let's assume we're estimating the opening degrees of valves
    return estimated_actions
