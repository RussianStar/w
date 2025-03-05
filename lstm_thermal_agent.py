#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LSTM-based Reinforcement Learning for Thermal System Control

This script implements a custom LSTM-based neural network for reinforcement learning
with the ThermalGymEnv environment. It uses PyTorch for GPU-accelerated training
and provides progress tracking during training.

Note: This script is designed to run on a specialized workstation with NVIDIA GPUs.
"""

import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import random
import argparse
from environment import ThermalGymEnv

# Set random seeds for reproducibility
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Check if CUDA is available and set device accordingly
if torch.cuda.is_available():
    device_count = torch.cuda.device_count()
    print(f"Found {device_count} CUDA device(s)")
    for i in range(device_count):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
    
    # Use the first GPU by default
    device = torch.device("cuda:0")
    print(f"Using device: {device}")
else:
    device = torch.device("cpu")
    print("CUDA not available, using CPU")


class ReplayBuffer:
    """Experience replay buffer for storing and sampling transitions."""
    
    def __init__(self, capacity):
        """Initialize a replay buffer with the given capacity."""
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        """Add a transition to the buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Sample a batch of transitions from the buffer."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors and move to device
        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.FloatTensor(np.array(actions)).to(device)
        rewards = torch.FloatTensor(np.array(rewards).reshape(-1, 1)).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        dones = torch.FloatTensor(np.array(dones).reshape(-1, 1)).to(device)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        """Return the current size of the buffer."""
        return len(self.buffer)


class LSTMActorCritic(nn.Module):
    """LSTM-based Actor-Critic network for continuous action spaces."""
    
    def __init__(self, state_dim, action_dim, hidden_dim=128, lstm_layers=2):
        """
        Initialize the LSTM Actor-Critic network.
        
        Args:
            state_dim (int): Dimension of the state space
            action_dim (int): Dimension of the action space
            hidden_dim (int): Dimension of the hidden layers
            lstm_layers (int): Number of LSTM layers
        """
        super(LSTMActorCritic, self).__init__()
        
        # Shared feature extractor
        self.lstm = nn.LSTM(
            input_size=state_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True
        )
        
        # Actor network (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Sigmoid()  # Output in [0, 1] range for valve opening degrees
        )
        
        # Critic network (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Action log standard deviation (learnable)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Initialize hidden state
        self.hidden = None
        
    def reset_hidden_state(self, batch_size=1):
        """Reset the hidden state of the LSTM."""
        self.hidden = (
            torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(device),
            torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(device)
        )
    
    def forward(self, state, sequence_length=1):
        """
        Forward pass through the network.
        
        Args:
            state (torch.Tensor): State tensor of shape (batch_size, sequence_length, state_dim)
            sequence_length (int): Length of the sequence for LSTM processing
            
        Returns:
            tuple: (action_mean, action_std, value)
        """
        batch_size = state.size(0)
        
        # Reshape state for LSTM if needed
        if len(state.shape) == 2:
            state = state.unsqueeze(1)  # Add sequence dimension
        
        # Initialize hidden state if None
        if self.hidden is None or self.hidden[0].size(1) != batch_size:
            self.reset_hidden_state(batch_size)
        
        # Pass through LSTM
        lstm_out, self.hidden = self.lstm(state, self.hidden)
        
        # Extract the last time step output
        features = lstm_out[:, -1, :]
        
        # Actor output (action mean)
        action_mean = self.actor(features)
        
        # Action standard deviation
        action_std = torch.exp(self.log_std).expand_as(action_mean)
        
        # Critic output (state value)
        value = self.critic(features)
        
        return action_mean, action_std, value
    
    def get_action(self, state, deterministic=False):
        """
        Sample an action from the policy.
        
        Args:
            state (numpy.ndarray): State array
            deterministic (bool): If True, return the mean action
            
        Returns:
            numpy.ndarray: Sampled action
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action_mean, action_std, _ = self.forward(state_tensor)
            
            if deterministic:
                action = action_mean
            else:
                # Sample from Normal distribution
                normal = Normal(action_mean, action_std)
                action = normal.sample()
                action = torch.clamp(action, 0.0, 1.0)  # Clamp to [0, 1]
            
            return action.cpu().numpy().squeeze()
    
    def evaluate_actions(self, states, actions):
        """
        Evaluate actions given states.
        
        Args:
            states (torch.Tensor): Batch of states
            actions (torch.Tensor): Batch of actions
            
        Returns:
            tuple: (values, log_probs, entropy)
        """
        action_means, action_stds, values = self.forward(states)
        
        # Create Normal distributions
        normal = Normal(action_means, action_stds)
        
        # Calculate log probabilities
        log_probs = normal.log_prob(actions).sum(dim=-1, keepdim=True)
        
        # Calculate entropy
        entropy = normal.entropy().sum(dim=-1, keepdim=True)
        
        return values, log_probs, entropy


class LSTMPPOAgent:
    """Proximal Policy Optimization agent with LSTM policy."""
    
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=128,
        lstm_layers=2,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        update_epochs=10,
        mini_batch_size=64,
        multi_gpu=False
    ):
        """
        Initialize the PPO agent.
        
        Args:
            state_dim (int): Dimension of the state space
            action_dim (int): Dimension of the action space
            hidden_dim (int): Dimension of the hidden layers
            lstm_layers (int): Number of LSTM layers
            lr (float): Learning rate
            gamma (float): Discount factor
            gae_lambda (float): GAE lambda parameter
            clip_ratio (float): PPO clip ratio
            value_coef (float): Value loss coefficient
            entropy_coef (float): Entropy loss coefficient
            max_grad_norm (float): Maximum gradient norm
            update_epochs (int): Number of epochs for each update
            mini_batch_size (int): Mini-batch size
            multi_gpu (bool): Whether to use multiple GPUs
        """
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.mini_batch_size = mini_batch_size
        
        # Create the actor-critic network
        self.policy = LSTMActorCritic(state_dim, action_dim, hidden_dim, lstm_layers)
        
        # Move to device
        self.policy.to(device)
        
        # Use DataParallel for multi-GPU training if requested and available
        if multi_gpu and torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs for training")
            self.policy = nn.DataParallel(self.policy)
        
        # Create optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
    
    def get_action(self, state, deterministic=False):
        """Get an action from the policy."""
        return self.policy.get_action(state, deterministic)
    
    def compute_gae(self, rewards, values, next_values, dones):
        """
        Compute Generalized Advantage Estimation.
        
        Args:
            rewards (torch.Tensor): Batch of rewards
            values (torch.Tensor): Batch of values
            next_values (torch.Tensor): Batch of next values
            dones (torch.Tensor): Batch of done flags
            
        Returns:
            tuple: (advantages, returns)
        """
        # Initialize advantages
        advantages = torch.zeros_like(rewards)
        
        # Initialize gae
        gae = 0
        
        # Compute advantages in reverse order
        for t in reversed(range(len(rewards))):
            # If t is the last step, use next_value, otherwise use values[t+1]
            if t == len(rewards) - 1:
                next_value = next_values[t]
            else:
                next_value = values[t + 1]
            
            # Compute TD error
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            
            # Compute GAE
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            
            # Store advantage
            advantages[t] = gae
        
        # Compute returns
        returns = advantages + values
        
        return advantages, returns
    
    def update(self, states, actions, rewards, next_states, dones, next_values):
        """
        Update the policy using PPO.
        
        Args:
            states (torch.Tensor): Batch of states
            actions (torch.Tensor): Batch of actions
            rewards (torch.Tensor): Batch of rewards
            next_states (torch.Tensor): Batch of next states
            dones (torch.Tensor): Batch of done flags
            next_values (torch.Tensor): Batch of next values
            
        Returns:
            dict: Dictionary of loss metrics
        """
        # Get values and log probs of actions
        values, old_log_probs, _ = self.policy.evaluate_actions(states, actions)
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae(rewards, values, next_values, dones)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        total_loss = 0
        policy_loss = 0
        value_loss = 0
        entropy_loss = 0
        
        # Perform multiple epochs of updates
        for _ in range(self.update_epochs):
            # Generate random indices
            indices = torch.randperm(states.size(0))
            
            # Split indices into mini-batches
            for start_idx in range(0, states.size(0), self.mini_batch_size):
                # Get mini-batch indices
                idx = indices[start_idx:start_idx + self.mini_batch_size]
                
                # Get mini-batch data
                mb_states = states[idx]
                mb_actions = actions[idx]
                mb_advantages = advantages[idx]
                mb_returns = returns[idx]
                mb_old_log_probs = old_log_probs[idx]
                
                # Reset LSTM hidden state for each mini-batch
                if hasattr(self.policy, 'module'):  # For DataParallel
                    self.policy.module.reset_hidden_state(len(idx))
                else:
                    self.policy.reset_hidden_state(len(idx))
                
                # Evaluate actions
                mb_values, mb_new_log_probs, mb_entropy = self.policy.evaluate_actions(mb_states, mb_actions)
                
                # Compute ratio
                ratio = torch.exp(mb_new_log_probs - mb_old_log_probs)
                
                # Compute surrogate losses
                surrogate1 = ratio * mb_advantages
                surrogate2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * mb_advantages
                
                # Compute policy loss
                mb_policy_loss = -torch.min(surrogate1, surrogate2).mean()
                
                # Compute value loss
                mb_value_loss = nn.MSELoss()(mb_values, mb_returns)
                
                # Compute entropy loss
                mb_entropy_loss = -mb_entropy.mean()
                
                # Compute total loss
                mb_loss = mb_policy_loss + self.value_coef * mb_value_loss + self.entropy_coef * mb_entropy_loss
                
                # Update policy
                self.optimizer.zero_grad()
                mb_loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Accumulate losses
                total_loss += mb_loss.item()
                policy_loss += mb_policy_loss.item()
                value_loss += mb_value_loss.item()
                entropy_loss += mb_entropy_loss.item()
        
        # Compute average losses
        num_updates = self.update_epochs * (states.size(0) // self.mini_batch_size + 1)
        total_loss /= num_updates
        policy_loss /= num_updates
        value_loss /= num_updates
        entropy_loss /= num_updates
        
        return {
            'total_loss': total_loss,
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'entropy_loss': entropy_loss
        }
    
    def save(self, path):
        """Save the policy to the given path."""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save the policy
        if hasattr(self.policy, 'module'):  # For DataParallel
            torch.save(self.policy.module.state_dict(), path)
        else:
            torch.save(self.policy.state_dict(), path)
        
        print(f"Model saved to {path}")
    
    def load(self, path):
        """Load the policy from the given path."""
        # Load the policy
        if hasattr(self.policy, 'module'):  # For DataParallel
            self.policy.module.load_state_dict(torch.load(path, map_location=device))
        else:
            self.policy.load_state_dict(torch.load(path, map_location=device))
        
        print(f"Model loaded from {path}")


def train_lstm_ppo(
    env,
    agent,
    total_timesteps=1000000,
    rollout_length=2048,
    save_freq=10000,
    log_freq=1000,
    eval_freq=10000,
    save_path="models/lstm_ppo_thermal",
    log_path="logs/lstm_ppo_thermal",
    eval_episodes=5
):
    """
    Train the LSTM PPO agent.
    
    Args:
        env (gym.Env): Gym environment
        agent (LSTMPPOAgent): LSTM PPO agent
        total_timesteps (int): Total number of timesteps to train for
        rollout_length (int): Number of timesteps to collect before updating
        save_freq (int): Frequency of saving the model
        log_freq (int): Frequency of logging metrics
        eval_freq (int): Frequency of evaluating the agent
        save_path (str): Path to save the model
        log_path (str): Path to save the logs
        eval_episodes (int): Number of episodes to evaluate the agent
        
    Returns:
        LSTMPPOAgent: Trained agent
    """
    # Create directories
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    
    # Create tensorboard writer
    writer = SummaryWriter(log_path)
    
    # Initialize variables
    timestep = 0
    episode = 0
    best_eval_reward = float('-inf')
    
    # Start timer
    start_time = time.time()
    
    # Training loop
    while timestep < total_timesteps:
        # Reset environment and agent hidden state
        state, _ = env.reset()
        agent.policy.reset_hidden_state()
        
        # Initialize episode variables
        episode_reward = 0
        episode_length = 0
        episode_constraint_violations = 0
        
        # Collect rollout
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        # Rollout loop
        for t in range(rollout_length):
            # Get action
            action = agent.get_action(state)
            
            # Take step in environment
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store transition
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(float(done))
            
            # Update state
            state = next_state
            
            # Update episode variables
            episode_reward += reward
            episode_length += 1
            if not info.get("constraints_met", True):
                episode_constraint_violations += 1
            
            # Increment timestep
            timestep += 1
            
            # Check if episode is done
            if done:
                # Log episode metrics
                writer.add_scalar('train/episode_reward', episode_reward, episode)
                writer.add_scalar('train/episode_length', episode_length, episode)
                writer.add_scalar('train/constraint_violations', episode_constraint_violations, episode)
                
                # Print episode metrics
                if episode % 10 == 0:
                    print(f"Episode {episode} - Reward: {episode_reward:.2f}, Length: {episode_length}, "
                          f"Violations: {episode_constraint_violations}, Timestep: {timestep}/{total_timesteps}")
                
                # Reset environment and agent hidden state for next episode
                state, _ = env.reset()
                agent.policy.reset_hidden_state()
                
                # Reset episode variables
                episode_reward = 0
                episode_length = 0
                episode_constraint_violations = 0
                
                # Increment episode counter
                episode += 1
            
            # Check if we need to save the model
            if timestep % save_freq == 0:
                agent.save(f"{save_path}_{timestep}.pt")
            
            # Check if we need to evaluate the agent
            if timestep % eval_freq == 0:
                eval_reward = evaluate_agent(env, agent, eval_episodes)
                writer.add_scalar('eval/mean_reward', eval_reward, timestep)
                
                # Save best model
                if eval_reward > best_eval_reward:
                    best_eval_reward = eval_reward
                    agent.save(f"{save_path}_best.pt")
                    print(f"New best model with reward {eval_reward:.2f}")
            
            # Check if we've reached the total timesteps
            if timestep >= total_timesteps:
                break
        
        # Convert lists to tensors
        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.FloatTensor(np.array(actions)).to(device)
        rewards = torch.FloatTensor(np.array(rewards).reshape(-1, 1)).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        dones = torch.FloatTensor(np.array(dones).reshape(-1, 1)).to(device)
        
        # Get next values for GAE
        with torch.no_grad():
            agent.policy.reset_hidden_state(states.size(0))
            _, _, next_values = agent.policy.forward(next_states)
        
        # Update agent
        update_metrics = agent.update(states, actions, rewards, next_states, dones, next_values)
        
        # Log update metrics
        if timestep % log_freq < rollout_length:
            for key, value in update_metrics.items():
                writer.add_scalar(f'train/{key}', value, timestep)
            
            # Calculate training speed
            elapsed_time = time.time() - start_time
            steps_per_second = timestep / elapsed_time
            writer.add_scalar('train/steps_per_second', steps_per_second, timestep)
            
            # Print progress
            progress = timestep / total_timesteps * 100
            print(f"Progress: {progress:.2f}% ({timestep}/{total_timesteps}), "
                  f"Steps/s: {steps_per_second:.2f}, "
                  f"Loss: {update_metrics['total_loss']:.4f}")
    
    # Save final model
    agent.save(f"{save_path}_final.pt")
    
    # Close tensorboard writer
    writer.close()
    
    return agent


def evaluate_agent(env, agent, n_episodes=10):
    """
    Evaluate the agent on the environment.
    
    Args:
        env (gym.Env): Gym environment
        agent (LSTMPPOAgent): LSTM PPO agent
        n_episodes (int): Number of episodes to evaluate
        
    Returns:
        float: Mean episode reward
    """
    # Initialize variables
    episode_rewards = []
    
    # Evaluation loop
    for episode in range(n_episodes):
        # Reset environment and agent hidden state
        state, _ = env.reset()
        agent.policy.reset_hidden_state()
        
        # Initialize episode variables
        episode_reward = 0
        done = False
        
        # Episode loop
        while not done:
            # Get action
            action = agent.get_action(state, deterministic=True)
            
            # Take step in environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Update state and reward
            state = next_state
            episode_reward += reward
        
        # Store episode reward
        episode_rewards.append(episode_reward)
        
        # Print episode reward
        print(f"Eval Episode {episode + 1}/{n_episodes} - Reward: {episode_reward:.2f}")
    
    # Calculate mean episode reward
    mean_reward = np.mean(episode_rewards)
    print(f"Evaluation - Mean Reward: {mean_reward:.2f} ± {np.std(episode_rewards):.2f}")
    
    return mean_reward


def main(args):
    """Main function."""
    # Print arguments
    print("Arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    
    # Create environment
    env = ThermalGymEnv(args.config_file)
    
    # Get state and action dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    
    # Create agent
    agent = LSTMPPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=args.hidden_dim,
        lstm_layers=args.lstm_layers,
        lr=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_ratio=args.clip_ratio,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm,
        update_epochs=args.update_epochs,
        mini_batch_size=args.mini_batch_size,
        multi_gpu=args.multi_gpu
    )
    
    # Train agent
    if not args.eval_only:
        print("Starting training...")
        start_time = time.time()
        
        agent = train_lstm_ppo(
            env=env,
            agent=agent,
            total_timesteps=args.total_timesteps,
            rollout_length=args.rollout_length,
            save_freq=args.save_freq,
            log_freq=args.log_freq,
            eval_freq=args.eval_freq,
            save_path=args.save_path,
            log_path=args.log_path,
            eval_episodes=args.eval_episodes
        )
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
    
    # Load best model if available
    best_model_path = f"{args.save_path}_best.pt"
    if os.path.exists(best_model_path):
        agent.load(best_model_path)
    
    # Evaluate agent
    print("Evaluating agent...")
    evaluate_agent(env, agent, args.eval_episodes)


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="LSTM-based PPO for Thermal Control")
    
    # Environment arguments
    parser.add_argument("--config_file", type=str, default="config.json", help="Path to the configuration file")
    
    # Agent arguments
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension of the LSTM")
    parser.add_argument("--lstm_layers", type=int, default=2, help="Number of LSTM layers")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE lambda parameter")
    parser.add_argument("--clip_ratio", type=float, default=0.2, help="PPO clip ratio")
    parser.add_argument("--value_coef", type=float, default=0.5, help="Value loss coefficient")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Entropy loss coefficient")
    parser.add_argument("--max_grad_norm", type=float, default=0.5, help="Maximum gradient norm")
    parser.add_argument("--update_epochs", type=int, default=10, help="Number of epochs for each update")
    parser.add_argument("--mini_batch_size", type=int, default=64, help="Mini-batch size")
    
    # Training arguments
    parser.add_argument("--total_timesteps", type=int, default=1000000, help="Total number of timesteps to train for")
    parser.add_argument("--rollout_length", type=int, default=2048, help="Number of timesteps to collect before updating")
    parser.add_argument("--save_freq", type=int, default=10000, help="Frequency of saving the model")
    parser.add_argument("--log_freq", type=int, default=1000, help="Frequency of logging metrics")
    parser.add_argument("--eval_freq", type=int, default=10000, help="Frequency of evaluating the agent")
    parser.add_argument("--save_path", type=str, default="models/lstm_ppo_thermal", help="Path to save the model")
    parser.add_argument("--log_path", type=str, default="logs/lstm_ppo_thermal", help="Path to save the logs")
    parser.add_argument("--eval_episodes", type=int, default=5, help="Number of episodes to evaluate the agent")
    
    # GPU arguments
    parser.add_argument("--multi_gpu", action="store_true", help="Whether to use multiple GPUs")
    
    # Evaluation arguments
    parser.add_argument("--eval_only", action="store_true", help="Whether to only evaluate the agent")
    
    args = parser.parse_args()
    
    main(args)
