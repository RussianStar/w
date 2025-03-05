#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GPU Monitoring Tool for LSTM Training

This script monitors GPU usage during LSTM training, providing real-time
information about GPU utilization, memory usage, temperature, and power consumption.
It's designed to run alongside the training process to help optimize resource usage.

Usage:
    python monitor_gpu.py [--interval 5] [--log-file gpu_stats.csv]
"""

import os
import time
import argparse
import subprocess
import csv
import datetime
import sys
from collections import deque

def get_gpu_stats():
    """
    Get current GPU statistics using nvidia-smi.
    
    Returns:
        list: List of dictionaries containing GPU statistics
    """
    try:
        # Run nvidia-smi to get GPU statistics
        result = subprocess.run([
            'nvidia-smi',
            '--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True, check=True)
        
        # Parse the output
        gpu_stats = []
        for line in result.stdout.strip().split('\n'):
            values = [val.strip() for val in line.split(',')]
            
            if len(values) >= 8:
                gpu_stats.append({
                    'index': int(values[0]),
                    'name': values[1],
                    'utilization': float(values[2]),
                    'memory_used': float(values[3]),
                    'memory_total': float(values[4]),
                    'temperature': float(values[5]),
                    'power_draw': float(values[6]),
                    'power_limit': float(values[7])
                })
        
        return gpu_stats
    
    except (subprocess.SubprocessError, ValueError) as e:
        print(f"Error getting GPU statistics: {e}")
        return []

def print_gpu_stats(gpu_stats, history=None, interval=5):
    """
    Print GPU statistics to the console.
    
    Args:
        gpu_stats (list): List of dictionaries containing GPU statistics
        history (dict): Dictionary of deques containing historical GPU statistics
        interval (int): Interval between measurements in seconds
    """
    # Clear screen
    os.system('cls' if os.name == 'nt' else 'clear')
    
    # Print header
    print("=" * 80)
    print(f"GPU MONITORING - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    if not gpu_stats:
        print("No GPU statistics available.")
        return
    
    # Print GPU statistics
    for gpu in gpu_stats:
        print(f"GPU {gpu['index']}: {gpu['name']}")
        print("-" * 80)
        
        # Calculate memory usage percentage
        memory_percent = (gpu['memory_used'] / gpu['memory_total']) * 100
        
        # Print current statistics
        print(f"Utilization:  {gpu['utilization']:6.2f}% ", end="")
        print_bar(gpu['utilization'])
        
        print(f"Memory:       {gpu['memory_used']:6.2f} / {gpu['memory_total']:6.2f} MB ", end="")
        print_bar(memory_percent)
        
        print(f"Temperature:  {gpu['temperature']:6.2f}°C")
        print(f"Power:        {gpu['power_draw']:6.2f} / {gpu['power_limit']:6.2f} W")
        
        # Print historical statistics if available
        if history:
            gpu_history = history.get(gpu['index'])
            if gpu_history:
                print("\nHistory (last 5 minutes):")
                
                # Calculate average utilization
                avg_util = sum(gpu_history['utilization']) / len(gpu_history['utilization'])
                print(f"Avg Utilization: {avg_util:6.2f}%")
                
                # Calculate average memory usage
                avg_mem = sum(gpu_history['memory_used']) / len(gpu_history['memory_used'])
                print(f"Avg Memory:      {avg_mem:6.2f} MB")
                
                # Calculate average temperature
                avg_temp = sum(gpu_history['temperature']) / len(gpu_history['temperature'])
                print(f"Avg Temperature: {avg_temp:6.2f}°C")
                
                # Calculate average power draw
                avg_power = sum(gpu_history['power_draw']) / len(gpu_history['power_draw'])
                print(f"Avg Power:       {avg_power:6.2f} W")
        
        print()
    
    print("=" * 80)
    print(f"Press Ctrl+C to stop monitoring. Refreshing every {interval} seconds.")

def print_bar(percentage, width=50):
    """
    Print a progress bar representing a percentage.
    
    Args:
        percentage (float): Percentage to represent (0-100)
        width (int): Width of the progress bar in characters
    """
    filled_width = int(percentage / 100 * width)
    bar = '█' * filled_width + '░' * (width - filled_width)
    print(f"[{bar}] {percentage:.2f}%")

def monitor_gpus(interval=5, log_file=None, duration=None):
    """
    Monitor GPU statistics continuously.
    
    Args:
        interval (int): Interval between measurements in seconds
        log_file (str): Path to CSV file to log statistics
        duration (int): Duration to monitor in seconds (None for indefinite)
    """
    # Initialize history
    history = {}
    history_length = int(300 / interval)  # 5 minutes of history
    
    # Initialize CSV writer if log file is specified
    csv_file = None
    csv_writer = None
    
    if log_file:
        try:
            csv_file = open(log_file, 'w', newline='')
            csv_writer = csv.writer(csv_file)
            
            # Write header
            csv_writer.writerow([
                'timestamp', 'gpu_index', 'name', 'utilization', 
                'memory_used', 'memory_total', 'temperature', 
                'power_draw', 'power_limit'
            ])
        except IOError as e:
            print(f"Error opening log file: {e}")
            log_file = None
    
    # Start monitoring
    start_time = time.time()
    try:
        while True:
            # Get current time
            current_time = time.time()
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Get GPU statistics
            gpu_stats = get_gpu_stats()
            
            # Update history
            for gpu in gpu_stats:
                gpu_index = gpu['index']
                
                if gpu_index not in history:
                    history[gpu_index] = {
                        'utilization': deque(maxlen=history_length),
                        'memory_used': deque(maxlen=history_length),
                        'temperature': deque(maxlen=history_length),
                        'power_draw': deque(maxlen=history_length)
                    }
                
                history[gpu_index]['utilization'].append(gpu['utilization'])
                history[gpu_index]['memory_used'].append(gpu['memory_used'])
                history[gpu_index]['temperature'].append(gpu['temperature'])
                history[gpu_index]['power_draw'].append(gpu['power_draw'])
            
            # Print GPU statistics
            print_gpu_stats(gpu_stats, history, interval)
            
            # Log statistics if log file is specified
            if csv_writer:
                for gpu in gpu_stats:
                    csv_writer.writerow([
                        timestamp, gpu['index'], gpu['name'], gpu['utilization'],
                        gpu['memory_used'], gpu['memory_total'], gpu['temperature'],
                        gpu['power_draw'], gpu['power_limit']
                    ])
                csv_file.flush()
            
            # Check if duration has elapsed
            if duration and (current_time - start_time) >= duration:
                break
            
            # Sleep until next interval
            time.sleep(interval)
    
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user.")
    
    finally:
        # Close CSV file if open
        if csv_file:
            csv_file.close()
            print(f"GPU statistics logged to {log_file}")

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Monitor GPU usage during LSTM training")
    parser.add_argument("--interval", type=int, default=5,
                        help="Interval between measurements in seconds")
    parser.add_argument("--log-file", type=str, default=None,
                        help="Path to CSV file to log statistics")
    parser.add_argument("--duration", type=int, default=None,
                        help="Duration to monitor in seconds (default: indefinite)")
    
    args = parser.parse_args()
    
    # Check if NVIDIA GPUs are available
    try:
        subprocess.run(['nvidia-smi'], capture_output=True, check=True)
    except (subprocess.SubprocessError, FileNotFoundError):
        print("Error: NVIDIA GPUs not detected or nvidia-smi not available.")
        print("This script requires NVIDIA GPUs with the NVIDIA driver installed.")
        sys.exit(1)
    
    # Start monitoring
    monitor_gpus(args.interval, args.log_file, args.duration)
