# Import the json module for reading JSON files
import json
# Import numpy library for array operations
import numpy as np
# Import pyplot from matplotlib for plotting
import matplotlib.pyplot as plt
# Import Axes3D for 3D plotting capabilities
from mpl_toolkits.mplot3d import Axes3D


# Define the main function
def main():
    # Open and read the trajectories JSON file
    with open('~/catkin_ws/src/3d_human_detection/data/trajectories.json', 'r') as json_file:
        # Load the JSON data into a list
        traj_list = json.load(json_file)

    # Create a new figure with size 20x20 inches
    fig = plt.figure(figsize=(20, 20))
    # Set number of subplot rows to 2
    num_rows = 2
    # Set number of subplot columns to 5
    num_cols = 5

    # Loop through each trajectory with its index
    for idx, traj in enumerate(traj_list):
        # Calculate the row position for this subplot
        row = idx // num_cols
        # Calculate the column position for this subplot
        col = idx % num_cols
        
        # Add a 3D subplot at position idx+1
        ax = fig.add_subplot(num_rows, num_cols, idx + 1, projection='3d')
        
        # Convert the trajectory to a numpy array
        traj = np.array(traj)
        # Extract the first 90 points from the trajectory
        traj_first_90 = traj[:90]
        # Extract the remaining points (last 45) from the trajectory
        traj_last_45 = traj[90:]
        
        # Extract X coordinates from first 90 points
        x_first_90 = traj_first_90[:, 0]
        # Extract Y coordinates from first 90 points
        y_first_90 = traj_first_90[:, 1]
        # Extract Z coordinates from first 90 points
        z_first_90 = traj_first_90[:, 2]
        
        # Extract X coordinates from last 45 points
        x_last_45 = traj_last_45[:, 0]
        # Extract Y coordinates from last 45 points
        y_last_45 = traj_last_45[:, 1]
        # Extract Z coordinates from last 45 points
        z_last_45 = traj_last_45[:, 2]
        
        # Plot the first 90 points as a blue line
        ax.plot(x_first_90, y_first_90, z_first_90, label="Init sequence", color='blue', linewidth=2)
        # Plot the last 45 points as a red line
        ax.plot(x_last_45, y_last_45, z_last_45, label="Out sequence", color='red', linewidth=2)
        
        # Set the X axis label
        ax.set_xlabel('X')
        # Set the Y axis label
        ax.set_ylabel('Y')
        # Set the Z axis label
        ax.set_zlabel('Z')
        # Set the subplot title with trajectory number
        ax.set_title(f'Trajectory {idx + 1}')
        # Set the 3D view angle (elevation=33°, azimuth=133°)
        ax.view_init(elev=33, azim=133)

    # Adjust subplot layout to prevent overlapping
    plt.tight_layout()
    # Display the plot
    plt.show()



# Check if the script is run directly (not imported)
if __name__ == "__main__":
    # Call the main function
    main()
