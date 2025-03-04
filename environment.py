import gymnasium as gym
from gymnasium import spaces
import numpy as np
from simulation import ThermalSimulation


class ThermalGymEnv(gym.Env):
    """
    A Gym environment for thermal system control using reinforcement learning.
    
    This environment simulates a thermal system with multiple water pipes,
    where the agent controls the opening degrees of valves to maintain
    temperatures above a legal minimum while minimizing energy usage.
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self, config_file: str):
        """
        Initialize the thermal gym environment.
        
        Args:
            config_file (str): Path to the configuration JSON file
        """
        super(ThermalGymEnv, self).__init__()
        
        # Initialize the thermal simulation
        self.sim = ThermalSimulation(config_file)
        self.n_pipes = self.sim.system_parameters["number_of_pipes_n"]
        
        # Define action space: continuous values between 0.0 (closed) and 1.0 (fully open) for each pipe
        self.action_space = spaces.Box(
            low=0.0, 
            high=1.0, 
            shape=(self.n_pipes,), 
            dtype=np.float32
        )
        
        # Define observation space: temperatures for each pipe, end sensor temperature, plus external temperature
        # Assuming temperatures can range from 0 to 100 degrees Celsius
        self.observation_space = spaces.Box(
            low=0.0, 
            high=100.0, 
            shape=(self.n_pipes + 2,),  # +2 for end sensor and external temperature
            dtype=np.float32
        )
        
        # Communication parameters
        self.lora_period = self.sim.communication_parameters["lora_period_seconds"]
        self.comm_cost_factor = self.sim.communication_parameters["communication_cost_factor"]
        
        # Track the current time step
        self.current_step = 0
        
    def reset(self, seed=None, options=None):
        """
        Reset the environment to initial state.
        
        Args:
            seed (int, optional): Random seed
            options (dict, optional): Additional options
            
        Returns:
            tuple: (observation, info)
        """
        print(f"DEBUG [Environment]: Resetting environment with seed={seed}")
        super().reset(seed=seed)
        self.current_step = 0
        state = self.sim.reset()
        info = {}
        print(f"DEBUG [Environment]: Reset state: {state}")
        return state, info
        
    def step(self, action):
        """
        Take a step in the environment using the provided action.
        
        Args:
            action (numpy.ndarray): Array of opening degrees for each pipe valve
            
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        print(f"DEBUG [Environment]: Step {self.current_step} - Action: {action}")
        
        # Validate action
        assert self.action_space.contains(action), f"Action {action} is invalid!"
        
        # Apply action to simulation
        state = self.sim.step(action)
        print(f"DEBUG [Environment]: State after step: {state}")
        
        # Check if constraints are met
        constraints_met = self.sim.check_constraints()
        print(f"DEBUG [Environment]: Constraints met: {constraints_met}")
        
        # Calculate reward
        reward = self._calculate_reward(state, action, constraints_met)
        print(f"DEBUG [Environment]: Reward: {reward}")
        
        # Determine if episode is terminated
        terminated = not constraints_met  # End episode if thermal constraints are violated
        
        # Truncated is False as we don't have a time limit
        truncated = False
        
        # Increment step counter
        self.current_step += 1
        
        # Additional info
        info = {
            "temperatures": self.sim.temperatures,
            "end_sensor_temp": self.sim.end_sensor_temperature,
            "outside_temp": self.sim.T_outside,
            "constraints_met": constraints_met,
            "time_step": self.current_step
        }
        
        return state, reward, terminated, truncated, info
        
    def render(self, mode='human'):
        """
        Render the environment.
        
        Args:
            mode (str): Rendering mode
            
        Returns:
            None
        """
        if mode == 'human':
            print(f"Step: {self.current_step}")
            print(f"Temperatures: {self.sim.temperatures}")
            print(f"End Sensor Temperature: {self.sim.end_sensor_temperature}")
            print(f"Outside Temperature: {self.sim.T_outside}")
            print(f"Opening Degrees: {self.sim.opening_degrees}")
            print(f"Constraints Met: {self.sim.check_constraints()}")
            print("-" * 50)
        else:
            raise NotImplementedError(f"Render mode {mode} not implemented")
            
    def _calculate_reward(self, state, action, constraints_met):
        """
        Calculate the reward for the current state and action.
        
        The reward function considers:
        1. Temperature maintenance (penalty for approaching minimum temperature)
        2. Energy efficiency (penalty for high opening degrees)
        3. Communication cost (penalty for frequent communications)
        4. Constraint violation (large penalty if temperature falls below minimum)
        
        Args:
            state (numpy.ndarray): Current state observation
            action (numpy.ndarray): Action taken
            constraints_met (bool): Whether all constraints are met
            
        Returns:
            float: Calculated reward
        """
        # Extract pipe temperatures (excluding end sensor and outside temp)
        pipe_temperatures = state[:self.n_pipes]
        
        # Extract end sensor temperature
        end_sensor_temp = state[self.n_pipes]
        
        # Base reward
        reward = 0.0
        
        # Penalty for approaching minimum temperature (encourage safety margin)
        temp_margin = pipe_temperatures - self.sim.T_law
        temp_penalty = -np.sum(1.0 / (temp_margin + 0.1))  # Adding 0.1 to avoid division by zero
        
        # Additional penalty for end sensor temperature approaching minimum
        end_sensor_margin = end_sensor_temp - self.sim.T_law
        end_sensor_penalty = -1.0 / (end_sensor_margin + 0.1)
        
        # Penalty for energy usage (proportional to valve opening)
        energy_penalty = -np.sum(action)
        
        # Communication cost penalty (if applicable)
        comm_penalty = 0.0
        if self.current_step % self.lora_period == 0 and self.current_step > 0:
            comm_penalty = -self.comm_cost_factor
            
        # Combine penalties
        reward = temp_penalty * 0.4 + end_sensor_penalty * 0.2 + energy_penalty * 0.3 + comm_penalty
        
        # Large penalty for constraint violation
        if not constraints_met:
            reward -= 100.0
            
        return reward
