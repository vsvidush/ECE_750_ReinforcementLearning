import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Directory containing the CSV files
input_folder = r"C:\Users\chawl\Documents\Project_RL\deep_RL\baselines\baselines\HOPPER\HALFCHEETAH_ALGORITHMS\Architecture_64"

# List of specific CSV files to use
csv_files = ["ACKTR.csv", "DDPG.csv", "PPO.csv", "TRPO.csv"]

# Initialize a dictionary to store data for plotting
data_dict = {}

# Function for smoothing data
def smooth_data(df, window=50):
    """
    Smooth data using a rolling mean.
    Args:
    - df (pd.DataFrame): Input DataFrame with columns Steps, Average Return, Standard Error.
    - window (int): Window size for smoothing.
    Returns:
    - pd.DataFrame: Smoothed DataFrame.
    """
    df["Average Return"] = df["Average Return"].rolling(window, min_periods=1).mean()
    df["Standard Error"] = df["Standard Error"].rolling(window, min_periods=1).mean()
    return df

# Function to normalize data to 1 million steps
def normalize_to_million_steps(df):
    """
    Normalize data to cover up to 1 million steps.
    Args:
    - df (pd.DataFrame): Input DataFrame with columns Steps, Average Return, Standard Error.
    Returns:
    - pd.DataFrame: Normalized DataFrame with steps up to 1 million.
    """
    max_steps = 1000000
    step_interval = 1000
    normalized_steps = np.arange(0, max_steps + 1, step_interval)
    
    # Interpolate missing data
    interp_returns = np.interp(normalized_steps, df["Steps"], df["Average Return"], left=0, right=df["Average Return"].iloc[-1])
    interp_errors = np.interp(normalized_steps, df["Steps"], df["Standard Error"], left=0, right=df["Standard Error"].iloc[-1])

    # Create a new DataFrame with normalized data
    normalized_df = pd.DataFrame({
        "Steps": normalized_steps,
        "Average Return": interp_returns,
        "Standard Error": interp_errors
    })
    return normalized_df

# Loop through the specified files
for filename in csv_files:
    file_path = os.path.join(input_folder, filename)
    if os.path.exists(file_path):
        try:
            # Extract algorithm from filename
            algorithm = filename.replace(".csv", "")
        except ValueError:
            print(f"Skipping malformed file name: {filename}")
            continue

        # Read the CSV file
        try:
            # Skip the first row and read the data with assumed column names
            data = pd.read_csv(file_path, skiprows=1, names=["Steps", "Average Return", "Standard Error"])
            
            # Sort the data by Steps
            data = data.sort_values("Steps")
            
            # Normalize data to 1 million steps
            data = normalize_to_million_steps(data)
            
            # Smooth the data
            data = smooth_data(data)
            
            # Store the data for plotting, using algorithm as the key
            data_dict[algorithm] = data
        except Exception as e:
            print(f"Error reading file {filename}: {e}")
    else:
        print(f"File not found: {filename}")

# Plot the data
if data_dict:
    plt.figure(figsize=(12, 8))
    for label, df in sorted(data_dict.items()):
        # Plot the Average Return line
        plt.plot(df["Steps"], df["Average Return"], label=label)
        
        # Add the shaded region for Standard Error
        plt.fill_between(
            df["Steps"], 
            df["Average Return"] - df["Standard Error"], 
            df["Average Return"] + df["Standard Error"], 
            alpha=0.3
        )

    # Customize the plot
    plt.title("halfcheetah-v5 (Algorithm Comparison)")
    plt.xlabel("Steps")
    plt.ylabel("Average Return")
    plt.legend()
    plt.grid()
    plt.tight_layout()

    # Save the plot
    output_plot = "halfcheetah_v5_comparison_plot.png"
    plt.savefig(output_plot)
    plt.show()

    print(f"Plot saved as {output_plot}.")
else:
    print("No valid data files to plot.")