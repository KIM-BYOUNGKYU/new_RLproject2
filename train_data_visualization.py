import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the data from the uploaded CSV file
file_path = 'train_data/storage/25_GAE2/training_data.csv'
data = pd.read_csv(file_path)

# Determine the episodes by identifying when the timestep resets to 1
data['episode'] = (data['epi_timestep'] == 1).cumsum()

# Calculate the length of each episode
episode_lengths = data.groupby('episode')['epi_timestep'].max()
print(episode_lengths)
# Identify the episode with the longest length
longest_episode = episode_lengths.idxmax()

# Extract the state0_3 values for the longest episode
longest_episode_data = data[data['episode'] == longest_episode]

# Convert the state0_3 strings to numpy arrays
def extract_coordinates(state_str):
    values = re.findall(r"[-+]?\d*\.\d+|\d+", state_str)
    return np.array([float(values[0]), float(values[1]), float(values[2])])

longest_episode_data['state0_3_coords'] = longest_episode_data['state0_3'].apply(extract_coordinates)

# Separate the coordinates into X, Y, Z lists for proper 3D plotting
x_coords = [coords[0] for coords in longest_episode_data['state0_3_coords']]
y_coords = [coords[1] for coords in longest_episode_data['state0_3_coords']]
z_coords = [coords[2] for coords in longest_episode_data['state0_3_coords']]

# Plot the state0_3 coordinates for the longest episode
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x_coords, y_coords, z_coords)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('State0_3 Coordinates for Longest Episode')

plt.show()