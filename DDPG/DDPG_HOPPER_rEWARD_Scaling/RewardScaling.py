import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Directory containing the CSV files
input_folder = r"C:\Users\chawl\Documents\Project_RL\deep_RL\baselines\baselines\DDPG_HOPPER_rEWARD_Scaling"


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

# Loop through all files in the folder
for filename in os.listdir(input_folder):
    if filename.endswith(".csv"):  # Check for CSV files
        try:
            # Extract reward scaling value from filename
            algorithm, reward_scaling = filename.split("_")
            reward_scaling = reward_scaling.replace(".csv", "").replace("e", "e-")
            reward_scaling = float(reward_scaling)
        except ValueError:
            print(f"Skipping malformed file name: {filename}")
            continue

        # Read the CSV file
        file_path = os.path.join(input_folder, filename)
        try:
            # Skip the first row and read the data with assumed column names
            data = pd.read_csv(file_path, skiprows=1, names=["Steps", "Average Return", "Standard Error"])
            
            # Sort the data by Steps
            data = data.sort_values("Steps")
            
            # Smooth the data
            data = smooth_data(data)
            
            # Store the data for plotting
            data_dict[reward_scaling] = data
        except Exception as e:
            print(f"Error reading file {filename}: {e}")

# Plot the data
if data_dict:
    plt.figure(figsize=(12, 8))
    for reward_scaling, df in sorted(data_dict.items()):
        # Plot the Average Return line
        plt.plot(df["Steps"], df["Average Return"], label=f"Reward Scaling: {reward_scaling}")
        
        # Add the shaded region for Standard Error
        plt.fill_between(
            df["Steps"], 
            df["Average Return"] - df["Standard Error"], 
            df["Average Return"] + df["Standard Error"], 
            color="gray", 
            alpha=0.3
        )

    # Customize the plot
    plt.title("Hopper-v5(DDPG,Reward Scale,With Norm)")
    plt.xlabel("Steps")
    plt.ylabel("Average Return")
    plt.legend()
    plt.grid()
    plt.tight_layout()

    # Save the plot
    output_plot = "reward_scaling_plot_shaded.png"
    plt.savefig(output_plot)
    plt.show()

    print(f"Plot saved as {output_plot}.")
else:
    print("No valid data files to plot.")
