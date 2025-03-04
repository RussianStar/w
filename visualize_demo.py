"""
Demonstration script for the thermal system visualization tool.

This script demonstrates how to use the visualization tool to display
the evolution of the thermal system's state over time.
"""

import numpy as np
from environment import ThermalGymEnv
from visualizer import ThermalSystemVisualizer, collect_episode_data, run_live_visualization


def run_random_policy_demo(max_steps=50):
    """
    Run a demonstration with a random policy.
    
    Args:
        max_steps (int): Maximum number of steps to run
        
    Returns:
        None
    """
    print("Running random policy demonstration...")
    
    # Create environment
    env = ThermalGymEnv("config.json")
    
    # Run live visualization with random policy
    run_live_visualization(env, policy=None, max_steps=max_steps, interval=100)


def run_constant_policy_demo(opening_degree=0.5, max_steps=50):
    """
    Run a demonstration with a constant policy.
    
    Args:
        opening_degree (float): Constant opening degree for all valves
        max_steps (int): Maximum number of steps to run
        
    Returns:
        None
    """
    print(f"Running constant policy demonstration (opening_degree={opening_degree})...")
    
    # Create environment
    env = ThermalGymEnv("config.json")
    
    # Define constant policy
    def constant_policy(state):
        return np.ones(env.sim.n_pipes, dtype=np.float32) * opening_degree
    
    # Run live visualization with constant policy
    run_live_visualization(env, policy=constant_policy, max_steps=max_steps, interval=100)


def run_adaptive_policy_demo(max_steps=50):
    """
    Run a demonstration with an adaptive policy.
    
    This policy adjusts valve openings based on temperature:
    - If temperature is close to minimum, open valve more
    - If temperature is well above minimum, close valve to save energy
    
    Args:
        max_steps (int): Maximum number of steps to run
        
    Returns:
        None
    """
    print("Running adaptive policy demonstration...")
    
    # Create environment
    env = ThermalGymEnv("config.json")
    
    # Define adaptive policy
    def adaptive_policy(state):
        # Extract temperatures (all except the last element, which is outside temp)
        temperatures = state[:-1]
        
        # Get minimum legal temperature
        T_law = env.sim.T_law
        
        # Initialize action
        action = np.zeros(env.sim.n_pipes, dtype=np.float32)
        
        # Adjust valve openings based on temperature
        for i in range(env.sim.n_pipes):
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
        
        return action
    
    # Run live visualization with adaptive policy
    run_live_visualization(env, policy=adaptive_policy, max_steps=max_steps, interval=100)


def save_animation_demo(policy_type="adaptive", filename="episode.mp4", max_steps=50):
    """
    Demonstrate saving an animation of an episode.
    
    Args:
        policy_type (str): Type of policy to use ("random", "constant", or "adaptive")
        filename (str): Output filename (should end with .mp4 or .gif)
        max_steps (int): Maximum number of steps to run
        
    Returns:
        None
    """
    print(f"Saving {policy_type} policy animation to {filename}...")
    
    # Create environment
    env = ThermalGymEnv("config.json")
    
    # Define policy
    if policy_type == "random":
        policy = None  # Random policy
    elif policy_type == "constant":
        def policy(state):
            return np.ones(env.sim.n_pipes, dtype=np.float32) * 0.5
    elif policy_type == "adaptive":
        def policy(state):
            # Extract temperatures
            temperatures = state[:-1]
            T_law = env.sim.T_law
            action = np.zeros(env.sim.n_pipes, dtype=np.float32)
            
            for i in range(env.sim.n_pipes):
                temp_margin = temperatures[i] - T_law
                
                if temp_margin < 5.0:
                    action[i] = 0.8
                elif temp_margin < 10.0:
                    action[i] = 0.5
                else:
                    action[i] = 0.2
            
            print(f"DEBUG [Save Animation Policy]: Action: {action}, dtype: {action.dtype}")
            return action
    else:
        raise ValueError(f"Unknown policy type: {policy_type}")
    
    # Collect episode data
    states, actions, rewards = collect_episode_data(env, policy=policy, max_steps=max_steps)
    
    # Create visualizer
    visualizer = ThermalSystemVisualizer("config.json")
    
    # Save animation
    visualizer.save_episode_animation(states, actions, rewards, filename, fps=5)


if __name__ == "__main__":
    print("=== Thermal System Visualization Demo ===\n")
    
    # Run demonstrations
    #run_random_policy_demo(max_steps=100)
    
    #run_constant_policy_demo(opening_degree=0.4, max_steps=100)
    
    run_adaptive_policy_demo(max_steps=100)
    
    # Uncomment to save animations
    # save_animation_demo(policy_type="random", filename="random_policy.mp4")
    # save_animation_demo(policy_type="constant", filename="constant_policy.mp4")
    # save_animation_demo(policy_type="adaptive", filename="adaptive_policy.mp4")
