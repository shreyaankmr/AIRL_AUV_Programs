import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pysindy import SINDy
from pysindy.optimizers import STLSQ
from pysindy.feature_library import PolynomialLibrary
from bayes_opt import BayesianOptimization
from sklearn.metrics import mean_squared_error
from math import sqrt

# Load CSV data
csv_path = '/Users/mrshreyaank/Desktop/IISC_Internship_AIRL/PINN/zigzag_dconst.csv'
data = pd.read_csv(csv_path, usecols=['Time', 'u', 'v', 'r', 'delta_e', 'delta_r'])

# Extract the relevant data
u = data['u'].values  # Surge velocity in m/s
v = data['v'].values  # Sway velocity in m/s
r = data['r'].values  # Yaw rate in rad/s
delta_r = data['delta_r'].values  # Rudder angle command in radians
time = data['Time'].values

# Prepare input (states) and control
x_train = np.column_stack((u[:-1], v[:-1], r[:-1]))  # State variables
u_control = delta_r[:-1].reshape(-1, 1)  # Control input

# Next states (for prediction targets)
x_next = np.column_stack((u[1:], v[1:], r[1:]))

# Define the SINDy model
def train_sindy_model(x_train, x_next, u_control, lamda1):
    # Define polynomial feature library
    library = PolynomialLibrary(degree=3)
    
    # Define the optimizer
    optimizer = STLSQ(threshold=lamda1)
    
    # Create SINDy model
    model = SINDy(feature_library=library, optimizer=optimizer)
    
    # Fit model using states and control inputs
    model.fit(x_train, u=u_control, x_dot=x_next, t=0.01)
    
    return model

# Define the objective function for Bayesian optimization
def sindy_objective(lamda1):
    model = train_sindy_model(x_train, x_next, u_control, lamda1)
    predicted_x_next = model.predict(x_train, u=u_control)
    error = np.mean((predicted_x_next - x_next)**2)
    return -error

# Define the search space
pbounds = {'lamda1': (0.01, 0.5)}

# Perform Bayesian optimization
optimizer = BayesianOptimization(
    f=sindy_objective,
    pbounds=pbounds,
    random_state=1,
)

optimizer.maximize(
    init_points=2,
    n_iter=100,
)

# Extract the best parameters
best_params = optimizer.max['params']

# Train the final SINDy model with the best parameters
final_model = train_sindy_model(x_train, x_next, u_control, best_params['lamda1'])

# Make predictions
predicted_x_next = final_model.predict(x_train, u=u_control)

# Calculate MSE and RMSE for each state
mse_u = mean_squared_error(x_next[:, 0], predicted_x_next[:, 0])
rmse_u = sqrt(mse_u)
mse_v = mean_squared_error(x_next[:, 1], predicted_x_next[:, 1])
rmse_v = sqrt(mse_v)
mse_r = mean_squared_error(x_next[:, 2], predicted_x_next[:, 2])
rmse_r = sqrt(mse_r)

print(f'MSE for u: {mse_u:.4f}')
print(f'RMSE for u: {rmse_u:.4f}')
print(f'MSE for v: {mse_v:.4f}')
print(f'RMSE for v: {rmse_v:.4f}')
print(f'MSE for r: {mse_r:.4f}')
print(f'RMSE for r: {rmse_r:.4f}')

# Plot the true and predicted states

# Plot true vs predicted 'u' (surge velocity)
plt.figure(figsize=(10, 10))

plt.subplot(3, 1, 1)
plt.plot(time[1:], u[1:], label='True u', color='blue')
plt.plot(time[1:], predicted_x_next[:, 0], label='Predicted u', color='orange', linestyle='--')
plt.xlabel('Time')
plt.ylabel('u (Surge Velocity)')
plt.title('True vs Predicted Surge Velocity (u)')
plt.legend()

# Plot true vs predicted 'v' (sway velocity)
plt.subplot(3, 1, 2)
plt.plot(time[1:], v[1:], label='True v', color='blue')
plt.plot(time[1:], predicted_x_next[:, 1], label='Predicted v', color='orange', linestyle='--')
plt.xlabel('Time')
plt.ylabel('v (Sway Velocity)')
plt.title('True vs Predicted Sway Velocity (v)')
plt.legend()

# Plot true vs predicted 'r' (yaw rate)
plt.subplot(3, 1, 3)
plt.plot(time[1:], r[1:], label='True r', color='blue')
plt.plot(time[1:], predicted_x_next[:, 2], label='Predicted r', color='orange', linestyle='--')
plt.xlabel('Time')
plt.ylabel('r (Yaw Rate)')
plt.title('True vs Predicted Yaw Rate (r)')
plt.legend()

plt.tight_layout()
plt.show()