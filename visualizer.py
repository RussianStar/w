"""
Visualization tool for the thermal system simulation.

This module provides a visualization tool for displaying the evolution of the
thermal system's state over time. It can be used to visualize the results of
simulations or training episodes.
"""

import numpy as np
import matplotlib
# Use an interactive backend for GUI visualization
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
from typing import List, Tuple, Dict, Optional, Union
import json
import os


class ThermalSystemVisualizer:
    """
    A visualization tool for the thermal system simulation.
    
    This class provides methods for visualizing the state of the thermal system
    over time, including temperatures, valve openings, and other relevant metrics.
    """
    
    def __init__(self, config_file: str = "config.json"):
        """
        Initialize the visualizer with the system configuration.
        
        Args:
            config_file (str): Path to the configuration JSON file
        """
        # Load configuration
        with open(config_file, 'r') as f:
            self.config = json.load(f)
            
        # Extract key parameters
        self.n_pipes = self.config["system_parameters"]["number_of_pipes_n"]
        self.T_law = self.config["system_parameters"]["T_law"]
        self.T_outside = self.config["system_parameters"]["T_outside"]
        
        # Set up color map for pipes
        self.colors = plt.cm.tab10(np.linspace(0, 1, self.n_pipes))
        
        # Initialize figure and axes
        self.fig = None
        self.axes = None
        
    def setup_plot(self):
        """
        Set up the plot layout for visualization.
        
        Returns:
            tuple: (fig, axes) - matplotlib figure and axes objects
        """
        # Create figure with grid layout
        self.fig = plt.figure(figsize=(12, 8))
        gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1])
        
        # Temperature plot
        ax_temp = self.fig.add_subplot(gs[0, :])
        ax_temp.set_title('Pipe Temperatures Over Time')
        ax_temp.set_xlabel('Time Step')
        ax_temp.set_ylabel('Temperature (°C)')
        ax_temp.grid(True)
        
        # Add horizontal line for minimum legal temperature
        ax_temp.axhline(y=self.T_law, color='r', linestyle='--', alpha=0.7, 
                        label=f'Minimum Legal Temperature ({self.T_law}°C)')
        
        # Add horizontal line for outside temperature
        ax_temp.axhline(y=self.T_outside, color='k', linestyle=':', alpha=0.5,
                        label=f'Outside Temperature ({self.T_outside}°C)')
        
        # Valve opening plot
        ax_valve = self.fig.add_subplot(gs[1, 0])
        ax_valve.set_title('Valve Openings')
        ax_valve.set_xlabel('Pipe')
        ax_valve.set_ylabel('Opening Degree')
        ax_valve.set_ylim(0, 1)
        ax_valve.grid(True)
        
        # Reward plot
        ax_reward = self.fig.add_subplot(gs[1, 1])
        ax_reward.set_title('Cumulative Reward')
        ax_reward.set_xlabel('Time Step')
        ax_reward.set_ylabel('Reward')
        ax_reward.grid(True)
        
        self.axes = {
            'temperature': ax_temp,
            'valve': ax_valve,
            'reward': ax_reward
        }
        
        self.fig.tight_layout()
        return self.fig, self.axes
        
    def plot_episode(self, states: List[np.ndarray], actions: List[np.ndarray], 
                    rewards: List[float], show_animation: bool = True):
        """
        Plot the evolution of the system state over an episode.
        
        Args:
            states (List[np.ndarray]): List of state observations
            actions (List[np.ndarray]): List of actions taken
            rewards (List[float]): List of rewards received
            show_animation (bool): Whether to show an animation of the episode
            
        Returns:
            tuple: (fig, axes) - matplotlib figure and axes objects
        """
        print(f"DEBUG: plot_episode called with {len(states)} states, {len(actions)} actions, {len(rewards)} rewards")
        print(f"DEBUG: First state shape: {states[0].shape}, content: {states[0]}")
        print(f"DEBUG: First action shape: {actions[0].shape if len(actions) > 0 else 'N/A'}, content: {actions[0] if len(actions) > 0 else 'N/A'}")
        
        if self.fig is None or self.axes is None:
            self.setup_plot()
            
        # Handle the case where we have one more state than actions/rewards
        # This happens because we include the initial state before any actions are taken
        if len(states) == len(actions) + 1:
            plot_states = states
            # For plotting temperatures over time, we use all states
            time_steps_temp = np.arange(len(states))
            # For plotting rewards, we use only the time steps where actions were taken
            time_steps_reward = np.arange(len(actions))
            print(f"DEBUG: Using {len(plot_states)} states for temperature plot, {len(actions)} actions for reward plot")
        else:
            # If states and actions have the same length, use them as is
            plot_states = states
            time_steps_temp = np.arange(len(states))
            time_steps_reward = np.arange(len(rewards))
            print(f"DEBUG: States and actions have same length: {len(states)}")
            
        # Extract temperature data (all except the last element of each state, which is outside temp)
        temperatures = np.array([state[:-1] for state in plot_states])
        print(f"DEBUG: Extracted temperatures shape: {temperatures.shape}")
        print(f"DEBUG: Temperature data sample (first state): {temperatures[0]}")
        
        # Clear previous plots
        for ax in self.axes.values():
            ax.clear()
            
        # Set up temperature plot
        ax_temp = self.axes['temperature']
        ax_temp.set_title('Pipe Temperatures Over Time')
        ax_temp.set_xlabel('Time Step')
        ax_temp.set_ylabel('Temperature (°C)')
        ax_temp.grid(True)
        
        # Plot temperature for each pipe
        for i in range(self.n_pipes):
            ax_temp.plot(time_steps_temp, temperatures[:, i], color=self.colors[i], 
                        label=f'Pipe {i+1}')
            
        # Add horizontal line for minimum legal temperature
        ax_temp.axhline(y=self.T_law, color='r', linestyle='--', alpha=0.7, 
                        label=f'Minimum Legal Temperature ({self.T_law}°C)')
        
        # Add horizontal line for outside temperature
        ax_temp.axhline(y=self.T_outside, color='k', linestyle=':', alpha=0.5,
                        label=f'Outside Temperature ({self.T_outside}°C)')
        
        ax_temp.legend(loc='upper right')
        
        # Set up valve opening plot for the final state
        ax_valve = self.axes['valve']
        ax_valve.set_title('Final Valve Openings')
        ax_valve.set_xlabel('Pipe')
        ax_valve.set_ylabel('Opening Degree')
        ax_valve.set_ylim(0, 1)
        ax_valve.grid(True)
        
        # Plot final valve openings
        pipe_indices = np.arange(1, self.n_pipes + 1)
        ax_valve.bar(pipe_indices, actions[-1], color=self.colors)
        ax_valve.set_xticks(pipe_indices)
        
        # Set up reward plot
        ax_reward = self.axes['reward']
        ax_reward.set_title('Cumulative Reward')
        ax_reward.set_xlabel('Time Step')
        ax_reward.set_ylabel('Reward')
        ax_reward.grid(True)
        
        # Plot cumulative reward
        cumulative_rewards = np.cumsum(rewards)
        ax_reward.plot(time_steps_reward, cumulative_rewards, 'g-')
        
        self.fig.tight_layout()
        
        if show_animation:
            try:
                self.animate_episode(states, actions, rewards)
            except Exception as e:
                print(f"WARNING: Could not show animation: {e}")
                print("Animation display is disabled. Use save_episode_animation to save animations to files.")
            
        return self.fig, self.axes
        
    def animate_episode(self, states: List[np.ndarray], actions: List[np.ndarray], 
                       rewards: List[float], interval: int = 200):
        """
        Create an animation of the episode.
        
        Args:
            states (List[np.ndarray]): List of state observations
            actions (List[np.ndarray]): List of actions taken
            rewards (List[float]): List of rewards received
            interval (int): Time interval between frames in milliseconds
            
        Returns:
            FuncAnimation: Matplotlib animation object
        """
        if self.fig is None or self.axes is None:
            self.setup_plot()
            
        # Handle the case where we have one more state than actions/rewards
        # This happens because we include the initial state before any actions are taken
        if len(states) == len(actions) + 1:
            # For animation, we need to pad the actions and rewards arrays
            # We'll duplicate the first action/reward for the initial state
            if len(actions) > 0:
                padded_actions = [actions[0]] + actions
                padded_rewards = [rewards[0]] + rewards
            else:
                # Handle edge case of empty actions/rewards
                padded_actions = [np.zeros(self.n_pipes)] * len(states)
                padded_rewards = [0.0] * len(states)
                
            animation_states = states
            animation_actions = padded_actions
            animation_rewards = padded_rewards
        else:
            # If states and actions have the same length, use them as is
            animation_states = states
            animation_actions = actions
            animation_rewards = rewards
            
        # Extract temperature data
        temperatures = np.array([state[:-1] for state in animation_states])
        time_steps = np.arange(len(animation_states))
        
        # Set up temperature plot
        ax_temp = self.axes['temperature']
        ax_temp.clear()
        ax_temp.set_title('Pipe Temperatures Over Time')
        ax_temp.set_xlabel('Time Step')
        ax_temp.set_ylabel('Temperature (°C)')
        ax_temp.grid(True)
        
        # Add horizontal line for minimum legal temperature
        ax_temp.axhline(y=self.T_law, color='r', linestyle='--', alpha=0.7, 
                       label=f'Minimum Legal Temperature ({self.T_law}°C)')
        
        # Add horizontal line for outside temperature
        ax_temp.axhline(y=self.T_outside, color='k', linestyle=':', alpha=0.5,
                       label=f'Outside Temperature ({self.T_outside}°C)')
        
        # Initialize temperature lines
        temp_lines = []
        for i in range(self.n_pipes):
            line, = ax_temp.plot([], [], color=self.colors[i], label=f'Pipe {i+1}')
            temp_lines.append(line)
            
        ax_temp.legend(loc='upper right')
        
        # Set temperature plot limits
        ax_temp.set_xlim(0, len(animation_states) - 1)
        min_temp = min(np.min(temperatures), self.T_outside, self.T_law) - 5
        max_temp = np.max(temperatures) + 5
        ax_temp.set_ylim(min_temp, max_temp)
        
        # Set up valve opening plot
        ax_valve = self.axes['valve']
        ax_valve.clear()
        ax_valve.set_title('Valve Openings')
        ax_valve.set_xlabel('Pipe')
        ax_valve.set_ylabel('Opening Degree')
        ax_valve.set_ylim(0, 1)
        ax_valve.grid(True)
        
        # Initialize valve bars
        pipe_indices = np.arange(1, self.n_pipes + 1)
        valve_bars = ax_valve.bar(pipe_indices, np.zeros(self.n_pipes), color=self.colors)
        ax_valve.set_xticks(pipe_indices)
        
        # Set up reward plot
        ax_reward = self.axes['reward']
        ax_reward.clear()
        ax_reward.set_title('Cumulative Reward')
        ax_reward.set_xlabel('Time Step')
        ax_reward.set_ylabel('Reward')
        ax_reward.grid(True)
        
        # Initialize reward line
        reward_line, = ax_reward.plot([], [], 'g-')
        
        # Set reward plot limits
        ax_reward.set_xlim(0, len(animation_states) - 1)
        cumulative_rewards = np.cumsum(animation_rewards)
        ax_reward.set_ylim(min(0, np.min(cumulative_rewards) - 10), 
                          max(0, np.max(cumulative_rewards) + 10))
        
        # Add time step indicator
        time_indicator = ax_temp.axvline(x=0, color='gray', linestyle='-', alpha=0.5)
        
        # Add text annotation for current time step
        time_text = ax_temp.text(0.02, 0.95, '', transform=ax_temp.transAxes)
        
        self.fig.tight_layout()
        
        def init():
            """Initialize animation"""
            for line in temp_lines:
                line.set_data([], [])
                
            for bar in valve_bars:
                bar.set_height(0)
                
            reward_line.set_data([], [])
            time_indicator.set_xdata([0])
            time_text.set_text('')
            
            return temp_lines + list(valve_bars) + [reward_line, time_indicator, time_text]
            
        def update(frame):
            """Update animation for each frame"""
            # Update temperature lines
            for i, line in enumerate(temp_lines):
                line.set_data(time_steps[:frame+1], temperatures[:frame+1, i])
                
            # Update valve bars
            for i, bar in enumerate(valve_bars):
                bar.set_height(animation_actions[frame][i])
                
            # Update reward line
            reward_line.set_data(time_steps[:frame+1], np.cumsum(animation_rewards[:frame+1]))
            
            # Update time indicator
            time_indicator.set_xdata([frame])
            
            # Update time text
            time_text.set_text(f'Time Step: {frame}')
            
            return temp_lines + list(valve_bars) + [reward_line, time_indicator, time_text]
            
        # Create animation
        anim = FuncAnimation(self.fig, update, frames=len(states),
                            init_func=init, blit=True, interval=interval)
        
        plt.show()
        
        return anim
        
    def save_episode_animation(self, states: List[np.ndarray], actions: List[np.ndarray], 
                              rewards: List[float], filename: str, fps: int = 5):
        """
        Save an animation of the episode to a file.
        
        Args:
            states (List[np.ndarray]): List of state observations
            actions (List[np.ndarray]): List of actions taken
            rewards (List[float]): List of rewards received
            filename (str): Output filename (should end with .mp4 or .gif)
            fps (int): Frames per second
            
        Returns:
            None
        """
        if self.fig is None or self.axes is None:
            self.setup_plot()
            
        # Create animation
        anim = self.animate_episode(states, actions, rewards, interval=1000//fps)
        
        # Save animation
        if filename.endswith('.mp4'):
            writer = 'ffmpeg'
        elif filename.endswith('.gif'):
            writer = 'pillow'
        else:
            raise ValueError("Filename should end with .mp4 or .gif")
            
        print(f"Saving animation to {filename}...")
        anim.save(filename, writer=writer, fps=fps)
        print(f"Animation saved to {filename}")


def collect_episode_data(env, policy=None, max_steps=100):
    """
    Collect data from an episode for visualization.
    
    Args:
        env: The environment to collect data from
        policy: Policy function to generate actions (if None, random actions are used)
        max_steps (int): Maximum number of steps to collect
        
    Returns:
        tuple: (states, actions, rewards) - Lists of states, actions, and rewards
    """
    states = []
    actions = []
    rewards = []
    
    # Reset the environment
    state, _ = env.reset()
    states.append(state)
    
    terminated = False
    truncated = False
    step = 0
    
    while not (terminated or truncated) and step < max_steps:
        # Generate action
        if policy is None:
            action = env.action_space.sample()
        else:
            action = policy(state)
            
        # Apply action
        next_state, reward, terminated, truncated, info = env.step(action)
        
        # Store data
        actions.append(action)
        rewards.append(reward)
        states.append(next_state)
        
        # Update state
        state = next_state
        step += 1
        
    return states, actions, rewards


def run_live_visualization(env, policy=None, max_steps=100, interval=100):
    """
    Run a live visualization of the environment with the given policy.
    
    This function runs the simulation and updates the visualization in real-time,
    rather than collecting all data first and then visualizing it afterward.
    
    Args:
        env: The environment to run
        policy: Policy function to generate actions (if None, random actions are used)
        max_steps (int): Maximum number of steps to run
        interval (int): Time interval between frames in milliseconds
        
    Returns:
        None
    """
    # Create visualizer
    visualizer = ThermalSystemVisualizer("config.json")
    
    # Initialize data storage
    states = []
    actions = []
    rewards = []
    
    # Reset the environment
    state, _ = env.reset()
    states.append(state)
    
    # Set up the initial plot
    fig, axes = visualizer.setup_plot()
    
    # Set up temperature plot
    ax_temp = axes['temperature']
    
    # Initialize temperature lines
    temp_lines = []
    for i in range(visualizer.n_pipes):
        line, = ax_temp.plot([], [], color=visualizer.colors[i], label=f'Pipe {i+1}')
        temp_lines.append(line)
        
    ax_temp.legend(loc='upper right')
    
    # Set temperature plot limits
    ax_temp.set_xlim(0, max_steps)
    min_temp = min(visualizer.T_outside, visualizer.T_law) - 5
    max_temp = 65  # Assuming initial temperature is around 60
    ax_temp.set_ylim(min_temp, max_temp)
    
    # Set up valve opening plot
    ax_valve = axes['valve']
    
    # Initialize valve bars
    pipe_indices = np.arange(1, visualizer.n_pipes + 1)
    valve_bars = ax_valve.bar(pipe_indices, np.zeros(visualizer.n_pipes), color=visualizer.colors)
    ax_valve.set_xticks(pipe_indices)
    
    # Set up reward plot
    ax_reward = axes['reward']
    
    # Initialize reward line
    reward_line, = ax_reward.plot([], [], 'g-')
    
    # Set reward plot limits
    ax_reward.set_xlim(0, max_steps)
    ax_reward.set_ylim(-100, 10)  # Will be adjusted as we get more data
    
    # Add time step indicator
    time_indicator = ax_temp.axvline(x=0, color='gray', linestyle='-', alpha=0.5)
    
    # Add text annotation for current time step
    time_text = ax_temp.text(0.02, 0.95, '', transform=ax_temp.transAxes)
    
    plt.ion()  # Turn on interactive mode
    plt.show()
    
    # Run the simulation and update the visualization in real-time
    terminated = False
    truncated = False
    step = 0
    
    try:
        while not (terminated or truncated) and step < max_steps:
            # Generate action
            if policy is None:
                action = env.action_space.sample()
            else:
                action = policy(state)
                
            # Apply action
            next_state, reward, terminated, truncated, info = env.step(action)
            
            # Store data
            actions.append(action)
            rewards.append(reward)
            states.append(next_state)
            
            # Update state
            state = next_state
            step += 1
            
            # Extract temperature data
            temperatures = np.array([s[:-1] for s in states])
            time_steps = np.arange(len(states))
            
            # Update temperature lines
            for i, line in enumerate(temp_lines):
                line.set_data(time_steps, temperatures[:, i])
                
            # Update valve bars
            for i, bar in enumerate(valve_bars):
                bar.set_height(action[i])
                
            # Update reward line
            cumulative_rewards = np.cumsum(rewards)
            reward_line.set_data(time_steps[:-1], cumulative_rewards)
            
            # Update time indicator
            time_indicator.set_xdata([step])
            
            # Update time text
            time_text.set_text(f'Time Step: {step}')
            
            # Adjust temperature plot x-limits if needed
            if step > ax_temp.get_xlim()[1]:
                ax_temp.set_xlim(0, step + 10)
                ax_reward.set_xlim(0, step + 10)
            
            # Adjust temperature plot y-limits if needed
            min_temp = min(np.min(temperatures), visualizer.T_outside, visualizer.T_law) - 5
            max_temp = max(np.max(temperatures), 65)  # Assuming initial temperature is around 60
            ax_temp.set_ylim(min_temp, max_temp)
            
            # Adjust reward plot y-limits if needed
            if len(cumulative_rewards) > 0:
                min_reward = min(0, np.min(cumulative_rewards) - 10)
                max_reward = max(0, np.max(cumulative_rewards) + 10)
                ax_reward.set_ylim(min_reward, max_reward)
            
            # Redraw the figure
            fig.canvas.draw()
            plt.pause(interval / 1000.0)
            
    except KeyboardInterrupt:
        print("Simulation interrupted by user")
    
    # Keep the plot open until the user closes it
    plt.ioff()  # Turn off interactive mode
    plt.show()


if __name__ == "__main__":
    # Example usage
    from environment import ThermalGymEnv
    
    # Create environment
    env = ThermalGymEnv("config.json")
    
    # Collect episode data with random policy
    states, actions, rewards = collect_episode_data(env)
    
    # Create visualizer
    visualizer = ThermalSystemVisualizer("config.json")
    
    # Visualize episode
    visualizer.plot_episode(states, actions, rewards)
    
    # Save animation
    # visualizer.save_episode_animation(states, actions, rewards, "episode.mp4")
