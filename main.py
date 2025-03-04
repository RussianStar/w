import gymnasium as gym
import numpy as np
from environment import ThermalGymEnv


def simple_test(episodes=3, max_steps=100):
    """
    Run a simple test of the thermal environment with random actions.
    
    Args:
        episodes (int): Number of episodes to run
        max_steps (int): Maximum steps per episode
        
    Returns:
        None
    """
    # Create the environment
    env = ThermalGymEnv("config.json")
    
    for episode in range(episodes):
        print(f"\n=== Episode {episode + 1} ===\n")
        
        # Reset the environment
        state, _ = env.reset()
        print(f"Initial State: {state}")
        print(f"Initial Temperatures: {state[:-1]}")
        print(f"Outside Temperature: {state[-1]}")
        
        # Run one episode
        total_reward = 0
        for step in range(max_steps):
            # Take a random action
            action = env.action_space.sample()
            print(f"\nStep {step + 1}")
            print(f"Action (valve openings): {action}")
            
            # Apply the action
            state, reward, terminated, truncated, info = env.step(action)
            
            # Render the environment
            env.render()
            
            # Accumulate reward
            total_reward += reward
            
            # Check if episode is done
            if terminated or truncated:
                print(f"\nEpisode ended after {step + 1} steps due to constraint violation!")
                break
                
        print(f"\nEpisode {episode + 1} completed with total reward: {total_reward:.2f}")
        
    print("\nTesting completed!")


def constant_action_test(opening_degree=0.5, steps=50):
    """
    Test the environment with a constant action for all valves.
    
    Args:
        opening_degree (float): Constant opening degree for all valves
        steps (int): Number of steps to run
        
    Returns:
        None
    """
    # Create the environment
    env = ThermalGymEnv("config.json")
    
    # Reset the environment
    state, _ = env.reset()
    print(f"Initial State: {state}")
    
    # Create constant action
    n_pipes = env.action_space.shape[0]  # Get the correct action shape from the action space
    action = np.ones(n_pipes, dtype=np.float32) * opening_degree  # Ensure float32 dtype to match action_space
    
    print(f"\n=== Constant Action Test (opening_degree={opening_degree}) ===\n")
    
    # Run for specified steps
    total_reward = 0
    for step in range(steps):
        print(f"\nStep {step + 1}")
        print(f"Action (valve openings): {action}")
        
        # Apply the action
        state, reward, terminated, truncated, info = env.step(action)
        
        # Render the environment
        env.render()
        
        # Accumulate reward
        total_reward += reward
        
        # Check if episode is done
        if terminated or truncated:
            print(f"\nTest ended after {step + 1} steps due to constraint violation!")
            break
            
    print(f"\nConstant action test completed with total reward: {total_reward:.2f}")


if __name__ == "__main__":
    print("=== Running Simple Random Action Test ===")
    simple_test(episodes=2, max_steps=20)
    
    print("\n\n=== Running Constant Action Tests ===")
    # Test with different constant opening degrees
    constant_action_test(opening_degree=0.3, steps=20)
    constant_action_test(opening_degree=0.7, steps=20)
