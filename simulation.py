import json
import numpy as np
import random


class ThermalSimulation:
    def __init__(self, config_file: str):
        """
        Initialize the thermal simulation with parameters from a config file.
        
        Args:
            config_file (str): Path to the configuration JSON file
        """
        self.config_file = config_file
        self.load_config(config_file)
        self.end_sensor_temperature = 0.0  # Initialize end sensor temperature
        self.reset()
        
    def load_config(self, config_file: str):
        """
        Load configuration parameters from a JSON file.
        
        Args:
            config_file (str): Path to the configuration JSON file
        """
        with open(config_file, 'r') as f:
            config = json.load(f)
            
        self.system_parameters = config["system_parameters"]
        self.pipes_config = config["pipes"]
        self.communication_parameters = config["communication_parameters"]
        self.initial_conditions = config["initial_conditions"]
        
        # Extract key parameters for easier access
        self.n_pipes = self.system_parameters["number_of_pipes_n"]
        self.c_water = self.system_parameters["c_water"]
        self.pump_power = self.system_parameters["pump_power"]
        self.T_outside = self.system_parameters["T_outside"]
        self.T_law = self.system_parameters["T_law"]
        
    def reset(self):
        """
        Reset the simulation to initial conditions.
        
        Returns:
            numpy.ndarray: The initial state of the system
        """
        # Initialize temperatures for each pipe
        self.temperatures = np.ones(self.n_pipes) * self.initial_conditions["initial_temperature"]
        
        # Initialize opening degrees for each pipe
        self.opening_degrees = np.array(self.initial_conditions["initial_opening_degrees"])
        
        # Initialize time step
        self.time_step = 0
        
        # Initialize end sensor temperature (weighted sum of pipe temperatures)
        self.update_end_sensor_temperature()
        
        return self.get_state()
        
    def update_end_sensor_temperature(self):
        """
        Update the end sensor temperature as the weighted sum of pipe temperatures.
        The weights are the valve opening percentages.
        """
        # If all valves are closed, use the average temperature
        if np.sum(self.opening_degrees) == 0:
            self.end_sensor_temperature = np.mean(self.temperatures)
            return
            
        # Calculate weighted sum of temperatures
        weighted_sum = np.sum(self.temperatures * self.opening_degrees)
        total_opening = np.sum(self.opening_degrees)
        
        # Normalize by total opening to get weighted average
        self.end_sensor_temperature = weighted_sum / total_opening
        
    def calculate_losses(self, pipe_idx, opening_degree):
        """
        Calculate heat losses for a specific pipe.
        
        Args:
            pipe_idx (int): Index of the pipe
            opening_degree (float): Opening degree of the pipe (0.0-1.0)
            
        Returns:
            float: Heat loss for the pipe
        """
        pipe_config = self.pipes_config[pipe_idx]
        
        # Constant loss based on temperature difference
        constant_loss = pipe_config["constant_loss_parameter"] * abs(self.temperatures[pipe_idx] - self.T_outside)
        
        # Stochastic loss (e.g., representing usage/extraction)
        stochastic_range = pipe_config["stochastic_loss_range"]
        stochastic_loss = random.uniform(stochastic_range[0], stochastic_range[1])
        
        # Calcification effect (reduces flow efficiency)
        calcification_factor = pipe_config["calcification_factor"]
        
        # Total loss calculation
        total_loss = constant_loss + stochastic_loss * opening_degree * (1 - calcification_factor)
        
        return total_loss
    
    def update_temperatures(self, opening_degrees):
        """
        Update temperatures based on opening degrees and heat losses.
        The model assumes that water is heated back up to the initial temperature
        after each cycle, and all valves are in parallel.
        
        Args:
            opening_degrees (list): List of opening degrees for each pipe
            
        Returns:
            numpy.ndarray: Updated temperatures
        """
        # Get the initial temperature (water is heated back up to this temperature)
        initial_temp = self.initial_conditions["initial_temperature"]
        
        # Create new temperatures array starting from the initial temperature
        new_temperatures = np.ones(self.n_pipes) * initial_temp
        
        for i in range(self.n_pipes):
            # Calculate water mass based on opening degree and pump power
            m_water = opening_degrees[i] * self.pump_power
            
            # Skip calculation if valve is completely closed
            if m_water <= 0:
                continue

            # Calculate heat loss for this pipe
            delta_q = self.calculate_losses(i, opening_degrees[i])
            
            # Calculate temperature change
            # ΔT = Δq / (c_water * m_water)
            delta_T = delta_q / (self.c_water * m_water) if m_water > 0 else 0
            # Update temperature (cooling effect from initial temperature)
            new_temperatures[i] -= delta_T
            
        return new_temperatures
        
    def step(self, opening_degrees):
        """
        Perform one simulation step with the given opening degrees.
        
        Args:
            opening_degrees (list): List of opening degrees for each pipe
            
        Returns:
            numpy.ndarray: The new state after the step
        """
        print(f"DEBUG [Simulation]: Step {self.time_step} - Opening degrees: {opening_degrees}")
        print(f"DEBUG [Simulation]: Current temperatures before update: {self.temperatures}")
        
        # Ensure opening degrees are within bounds [0, 1]
        opening_degrees = np.clip(opening_degrees, 0.0, 1.0)
        
        # Update temperatures based on opening degrees
        self.temperatures = self.update_temperatures(opening_degrees)
        
        print(f"DEBUG [Simulation]: Updated temperatures: {self.temperatures}")
        
        # Store current opening degrees
        self.opening_degrees = opening_degrees
        
        # Update end sensor temperature
        self.update_end_sensor_temperature()
        print(f"DEBUG [Simulation]: End sensor temperature: {self.end_sensor_temperature}")
        
        # Increment time step
        self.time_step += 1
        
        state = self.get_state()
        print(f"DEBUG [Simulation]: Returned state: {state}")
        return state
        
    def get_state(self):
        """
        Get the current state of the system.
        
        Returns:
            numpy.ndarray: Current state (pipe temperatures, end sensor temperature, and external temperature)
        """
        # State includes all pipe temperatures, end sensor temperature, and the outside temperature
        state = np.append(self.temperatures, [self.end_sensor_temperature, self.T_outside])
        return state
        
    def check_constraints(self):
        """
        Check if all thermal constraints are met.
        
        Returns:
            bool: True if all constraints are met, False otherwise
        """
        # Check if all pipe temperatures and end sensor temperature are above the legal minimum (T_law)
        return np.all(self.temperatures >= self.T_law) and self.end_sensor_temperature >= self.T_law
