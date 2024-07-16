import pandas as pd
import numpy as np
import pysindy as ps
import matplotlib.pyplot as plt

# Step 1: Load and preprocess data
data = pd.read_csv('/Users/mrshreyaank/Desktop/IISC_Internship_AIRL/mss_data.csv')

# Extract relevant columns for states and controls
states = data[['X', 'Y', 'Z', 'Phi', 'Theta', 'Psi', 'U', 'V', 'W', 'P', 'Q', 'R']]
controls = data[['delta_r', 'delta_s', 'n_prop']]

# Step 2: Concatenate states and controls (excluding Time column)
states_with_controls = np.hstack((states.values, controls.values))

# Optionally, you can retain the Time column separately if needed
time_column = data['Time'].values

# Define the feature library to include states and control inputs
state_names = ['X', 'Y', 'Z', 'Phi', 'Theta', 'Psi', 'U', 'V', 'W', 'P', 'Q', 'R']
control_names = ['delta_r', 'delta_s', 'n_prop']
feature_names = state_names + control_names

# Create custom functions for the library
functions = [
    lambda : 1,
    lambda x: x,             # linear terms
    lambda x: x**2,          # quadratic terms
    lambda x: np.sin(x),     # sine terms
    lambda x: np.cos(x),     # cosine terms
    lambda x,y: x*y,
    lambda x: x*abs(x),
    lambda x,y: (x**2)*y,
    lambda x,y: x*np.sin(y),
    lambda x,y: x*np.cos(y),
    lambda x,y: np.cos(x)*np.sin(y),
    lambda x,y: np.cos(x)*np.sin(y)
    ]

# Define the custom library
library = ps.CustomLibrary(library_functions=functions, include_bias=True)

def fit_sindy_model(states, controls, threshold=1e-7, epochs=100000):
    # Define the optimizer with specified threshold and epochs
    optimizer = ps.STLSQ(threshold=threshold, max_iter=epochs)
    
    # Instantiate and fit the SINDy model with the optimizer and custom library
    model = ps.SINDy(optimizer=optimizer, feature_library=library, feature_names=feature_names)
    model.fit(np.hstack((states, controls)), t=0.02)  # Assuming sampleTime = 0.02
    return model

# Train SINDy model with default threshold and epochs
model = fit_sindy_model(states, controls)
model.print()

# Predict states using the trained SINDy model
predicted_states_controls = model.predict(np.hstack((states, controls)))

# Define state and control names for plotting
all_names = state_names + control_names

# Plotting the true and predicted states and controls
num_plots = len(all_names)
fig, axes = plt.subplots(nrows=num_plots // 3 + (num_plots % 3 > 0), ncols=3, figsize=(15, 10))
axes = axes.flatten()

for i, name in enumerate(all_names):
    if i < len(state_names):
        true_values = states[state_names[i]].values
    else:
        true_values = controls[control_names[i - len(state_names)]].values
    
    axes[i].plot(time_column, true_values, label='True')
    axes[i].plot(time_column, predicted_states_controls[:, i], label='Predicted', linestyle='--')
    axes[i].set_title(name)
    axes[i].legend()

for i in range(len(all_names), len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.show()