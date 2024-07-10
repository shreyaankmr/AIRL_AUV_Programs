import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import control as ctrl
from sklearn.metrics import mean_squared_error

# Known values for the system
m = 30.48                 # Mass of the vehicle
X_udot = -9.3e-1            # Added mass in the surge direction
Z_wdot = -35.5           # Added mass in the heave direction
M_qdot = -4.88           # Added mass moment of inertia in pitch
xg = 0.0                 # Longitudinal center of gravity
zg = 0.0196             # Vertical center of gravity
zb = 0.0
W = m*9.81
B = W + (0.75*4.44822162)               # Hydrodynamic damping force in surge due to hydrodynamic surfaces
X_prop = 3.86         # Thrust force in surge from the propeller
Z_qdot = -1.93
M_wdot = -1.93
xb = 0
Iyy = 3.45
cdu = 0.2;
rho = 1030;
Af = 0.0285;

# Parameters for the system
params = {
    'X_uu': -0.5*rho*cdu*Af,# -1.62,
    'X_wq': -35.5,
    'X_qq': -1.93,
    'Z_ww': -131.0,
    'Z_qq': -0.632,
    'Z_uq': -5.22,
    'Z_uw': -28.6,
    'Z_uu_delta_s': -9.64,
    'M_ww': 3.18,
    'M_qq': -188.0,
    'M_uq': -2.0,
    'M_uw': 24.0,
    'M_uu_delta_s': -6.15
}

# Define the derivatives
def remus_vehicle_numpy(t, x, u_input, params):
    u, w, q, z, theta = x
    delta_s = u_input[0]  # External control input
    
    X_HS = -(W - B) * np.sin(theta)
    Z_HS = (W - B)*np.cos(theta)
    M_HS = -(zg * W - zb * B) * np.sin(theta) - (xg * W - xb * B) * np.cos(theta)

    coeff = np.array([
        [m - X_udot, 0, m * zg, 0, 0],
        [0, m - Z_wdot, -(m * xg + Z_qdot), 0, 0],
        [m * zg, -(m * xg + M_wdot), Iyy - M_qdot, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1]
    ])
    coeff_inv = np.linalg.inv(coeff)
    summation = np.array([
        [X_HS + params['X_uu'] * u * abs(u) + (params['X_wq'] - m) * w * q + (params['X_qq'] + m * xg) * q**2 + X_prop],
        [Z_HS + params['Z_ww'] * w * abs(w) + params['Z_qq'] * q * abs(q) + (params['Z_uq'] + m) * u * q + params['Z_uw'] * u * w + params['Z_uu_delta_s'] * u**2 * delta_s],
        [M_HS + params['M_ww'] * w * abs(w) + params['M_qq'] * q * abs(q) + (params['M_uq'] - m * xg) * u * q - m * zg * w * q + params['M_uw'] * u * w + params['M_uu_delta_s'] * u**2 * delta_s],
        [-u * np.sin(theta) + w * np.cos(theta)],
        [q]
    ])
    dot = np.matmul(coeff_inv, summation)
    return dot.flatten()

    

# Initial conditions
initial_state = [1.5, 0.0, 0.0, 0.0, 0.0]
u0=1.5
w0=0.0
q0=0.0
z0=0.0
theta0=0.0
# Time array
t = np.arange(0, 10, 0.001)

# Define the control input
u_input = np.zeros_like(t)
u_input[1000:4000] = 0.04  # 1 to 4 seconds
u_input[4000:5000] = -0.04  # 4 to 5 seconds
u_input[5000:7000] = 0.02  # 5 to 7 seconds
u_input[7000:9000] = -0.02  # 7 to 9 seconds
u_input[9000:] = 0.0  # 9 to 20 seconds
u_input[:-1]=0
# Define the filter transfer function
numerator = [0.5]
denominator = [1, 0.5]
filter_tf = ctrl.TransferFunction(numerator, denominator)

# Apply the filter to the input
t_out, u_filtered = ctrl.forced_response(filter_tf, T=t, U=u_input)

# Create NonlinearIOSystem
nonlinear_system = ctrl.NonlinearIOSystem(
    updfcn=remus_vehicle_numpy,  # State derivative function
    inputs=1, outputs=5, states=5, params=params,
    name='remus_vehicle'
)

# Run the simulation
T, yout = ctrl.input_output_response(nonlinear_system, T=t, U=u_filtered, X0=initial_state)
plt.figure(figsize=(10, 6))
plt.plot(T, yout[0], label='u (x-direction velocity)')
plt.plot(T, yout[1], label='w (z-direction velocity)')
plt.title('REMUS Underwater Vehicle Simulation')
plt.grid(True)
plt.legend()

plt.figure(figsize=(10, 6))
plt.plot(T, yout[4], label='theta (pitch angle)')
plt.plot(T, yout[2], label='q (pitch rate)')
plt.title('REMUS Underwater Vehicle Simulation')
plt.grid(True)
plt.legend()

plt.figure(figsize=(10, 6))
plt.plot(T, u_input, label='Filtered stern angle')
plt.title('REMUS Underwater Vehicle Simulation')
plt.grid(True)
plt.legend()

plt.figure(figsize=(10, 6))
plt.plot(T, yout[3], label='z (z direction position)')
plt.xlabel('Time (s)')
plt.legend()
plt.title('REMUS Underwater Vehicle Simulation')
plt.grid(True)
plt.show()
