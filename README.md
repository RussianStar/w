py# Thermal System Simulation Gym

A reinforcement learning environment for optimizing the control of a thermal system with multiple water pipes. This project provides a simulation of a thermal system and a Gym environment for training RL agents to control valve openings while maintaining temperature constraints.

## Project Overview

This project simulates a thermal system with multiple parallel water pipes, each with its own valve that can be controlled to regulate flow. The goal is to maintain temperatures above a legal minimum while minimizing energy usage and communication costs.

### Key Features

- Configurable thermal system simulation with realistic physics
- OpenAI Gym-compatible environment for reinforcement learning
- Customizable reward function balancing temperature maintenance, energy efficiency, and communication costs
- Detailed state tracking and constraint checking
- Visualization of system state during simulation

## Installation

1. Clone this repository:
   ```
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Basic Usage

To run a simple test of the environment with random actions:

```python
from environment import ThermalGymEnv

# Create the environment
env = ThermalGymEnv("config.json")

# Reset the environment
state = env.reset()

# Run a few steps with random actions
for _ in range(10):
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)
    env.render()
    
    if done:
        break
```

### Running the Test Script

The project includes a test script that demonstrates the environment's functionality:

```
python main.py
```

This will run several test episodes with both random and constant actions.

## Configuration

The system is configured through a JSON file (`config.json`). Here's an explanation of the key parameters:

### System Parameters

- `number_of_pipes_n`: Number of parallel water pipes in the system
- `c_water`: Specific heat capacity of water (J/kg·K)
- `pump_power`: Pump power affecting water flow (kg/s at full valve opening)
- `T_outside`: Outside/ambient temperature (°C)
- `T_law`: Legal minimum temperature that must be maintained (°C)

### Pipe Configuration

Each pipe has individual parameters:

- `pipe_index`: Index of the pipe (0-based)
- `constant_loss_parameter`: Heat loss coefficient
- `stochastic_loss_range`: Range for random heat extraction [min, max]
- `calcification_factor`: Factor representing pipe efficiency reduction due to calcification

### Communication Parameters

- `lora_period_seconds`: Period between LoRa communications (in simulation steps)
- `communication_cost_factor`: Cost factor for each communication
- `max_allowed_fail_prob`: Maximum allowed probability of communication failure

### Initial Conditions

- `initial_temperature`: Initial temperature of all pipes (°C)
- `initial_opening_degrees`: Initial valve opening degrees for each pipe (0.0-1.0)

## Visualization Tool

The project includes a visualization tool for displaying the evolution of the thermal system's state over time. This tool can be used to visualize the results of simulations or training episodes.

### Basic Usage

```python
from environment import ThermalGymEnv
from visualizer import ThermalSystemVisualizer, collect_episode_data

# Create environment
env = ThermalGymEnv("config.json")

# Collect episode data with random policy
states, actions, rewards = collect_episode_data(env, max_steps=50)

# Create visualizer
visualizer = ThermalSystemVisualizer("config.json")

# Visualize episode
visualizer.plot_episode(states, actions, rewards)

# Save animation (optional)
visualizer.save_episode_animation(states, actions, rewards, "episode.mp4")
```

### Visualization Features

The visualization tool provides the following features:

- Temperature plot showing the evolution of temperatures for each pipe over time
- Valve opening plot showing the valve positions for each pipe
- Reward plot showing the cumulative reward over time
- Animation of the episode with a time slider
- Option to save animations as MP4 or GIF files

### Demo Script

The project includes a demonstration script (`visualize_demo.py`) that shows how to use the visualization tool with different policies:

- Random policy: Actions are randomly sampled from the action space
- Constant policy: All valves are set to the same opening degree
- Adaptive policy: Valve openings are adjusted based on temperature margins

Run the demo script to see the visualization tool in action:

```
python visualize_demo.py
```

## Extending the Project

### Adding Custom Agents

You can implement custom RL agents to control the environment. For example, using stable-baselines3:

```python
from stable_baselines3 import PPO
from environment import ThermalGymEnv

# Create environment
env = ThermalGymEnv("config.json")

# Create agent
model = PPO("MlpPolicy", env, verbose=1)

# Train agent
model.learn(total_timesteps=10000)

# Save agent
model.save("ppo_thermal")

# Test agent
obs, _ = env.reset()
terminated = truncated = False
while not (terminated or truncated):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
```

### Modifying the Reward Function

The reward function can be customized by modifying the `_calculate_reward` method in the `ThermalGymEnv` class. The current implementation balances:

1. Temperature maintenance (penalty for approaching minimum temperature)
2. Energy efficiency (penalty for high valve openings)
3. Communication costs (periodic penalty for LoRa communications)
4. Constraint violations (large penalty if temperature falls below minimum)

### Integrating Visualization with Training

The visualization tool can be integrated with the training process to monitor the agent's performance during training. For example:

```python
from stable_baselines3 import PPO
from environment import ThermalGymEnv
from visualizer import ThermalSystemVisualizer, collect_episode_data

# Create environment and agent
env = ThermalGymEnv("config.json")
model = PPO("MlpPolicy", env, verbose=1)

# Create visualizer
visualizer = ThermalSystemVisualizer("config.json")

# Training loop with visualization
for i in range(10):  # 10 training iterations
    # Train for some steps
    model.learn(total_timesteps=1000)
    
    # Evaluate and visualize
    eval_env = ThermalGymEnv("config.json")
    states, actions, rewards = [], [], []
    
    obs, _ = eval_env.reset()
    terminated = truncated = False
    while not (terminated or truncated):
        states.append(obs)
        action, _ = model.predict(obs, deterministic=True)
        actions.append(action)
        obs, reward, terminated, truncated, _ = eval_env.step(action)
        rewards.append(reward)
    
    # Visualize episode
    visualizer.plot_episode(states, actions, rewards)
```

## Mathematical Model

The thermal system is modeled using the following equations:

1. Heat loss calculation for each pipe:
   ```
   Δq_n ≈ |T_n - T_outside| · c_n + f_extraction
   ```

2. Temperature change calculation:
   ```
   ΔT_n = Δq_n / (c_water · m_water)
   ```
   where `m_water = opening_degree · pump_power`

## License

[Specify license information here]
