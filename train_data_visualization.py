import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the data from the uploaded CSV file
file_path = 'train_data/training_data.csv'
data = pd.read_csv(file_path, low_memory=False)

s_values = data[['S_0', 'S_1', 'S_2']].apply(pd.to_numeric, errors='coerce')

# Drop rows with NaN values
s_values = s_values.dropna()

# Creating a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot(s_values['S_0'], s_values['S_1'], s_values['S_2'], c='b', linewidth = 0.1)

ax.set_xlabel('S_0')
ax.set_ylabel('S_1')
ax.set_zlabel('S_2')

plt.show()

# Ensure that S_0, S_1, and S_2 columns contain only numeric data
data[['S_0', 'S_1', 'S_2']] = data[['S_0', 'S_1', 'S_2']].apply(pd.to_numeric, errors='coerce')

# Group by 'Epi' and find the Epi with the most entries
value_count= data.groupby('Epi').count()
most_common_epi =value_count['Timestep'].drop('Epi').idxmax()
print(most_common_epi)
# Filter the data to include only the rows with the most common Epi
filtered_data = data[data['Epi'] == most_common_epi]

# Drop rows with NaN values in S_0, S_1, and S_2 columns
filtered_data = filtered_data.dropna(subset=['S_0', 'S_1', 'S_2'])

# Extract S_0, S_1, and S_2 columns for the most common Epi
s_values = filtered_data[['S_0', 'S_1', 'S_2']]

# Creating a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot(s_values['S_0'], s_values['S_1'], s_values['S_2'], c='b', marker='o')

ax.set_xlabel('S_0')
ax.set_ylabel('S_1')
ax.set_zlabel('S_2')
ax.set_title(f'Epi {most_common_epi}')

plt.show()
