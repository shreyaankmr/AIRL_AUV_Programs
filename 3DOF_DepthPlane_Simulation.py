import numpy as np
import matplotlib.pyplot as plt
import control as ctrl

# Parameters for REMUS underwater vehicle (example values)
m = 30.48 #63.0                 # Mass of the vehicle
X_udot = -9.3e-1            # Added mass in the surge direction
Z_wdot = -35.5           # Added mass in the heave direction
M_qdot = -4.88           # Added mass moment of inertia in pitch
xg = 0.0                 # Longitudinal center of gravity
zg = 1.96e-2             # Vertical center of gravity
zb = 0.0
#I_yy = 20.0              # Moment of inertia about the y-axis
W = 2.99e2
B = 302.3449662#3.08e2               # Hydrodynamic damping force in surge due to hydrodynamic surfaces
#X_HS = -(W-B)
X_prop = 3.86*1          # Thrust force in surge from the propeller
Z_HS = 9.0               # Hydrodynamic damping force in heave due to hydrodynamic surfaces
M_HS = 2.0               # Hydrodynamic damping moment in pitch due to hydrodynamic surfaces
M_uu = 0.0               # Nonlinear damping moment coefficient in pitch
Z_uw = -2.86e1           # Cross-term drag coefficient between surge and heave
Z_uu_delta_s = -9.64     # Quadratic drag coefficient in heave due to control surface deflection
Z_w = 1.0                # Linear drag coefficient in heave
Z_q = -1.93              # Linear drag coefficient in pitch
M_uq = -2.0              # Cross-term moment coefficient between surge and pitch
M_uw = 2.4e1             # Cross-term moment coefficient between surge and heave
M_w = 1.0                # Nonlinear drag moment coefficient in heave
M_q = 1.0                # Nonlinear drag moment coefficient in pitch
Z_uq = -5.22             # Cross-term drag coefficient between surge and pitch
X_wq = -3.55e1           # Cross-term drag coefficient between heave and pitch
X_qq = -1.93             # Quadratic drag coefficient in pitch
X_uu_delta_s = -4.403    # Quadratic drag coefficient in surge due to control surface deflection
Z_qdot = -1.93
M_wdot = -1.93
X_uu = -4.40
M_uu_delta_s = -6.15
Z_ww = -1.31e2
Z_qq = -6.32e-1
M_ww = 3.18
M_qq = -1.88e2
xb = 0#4.37e-1
X_uu = -1.62
u_prev = 0.0  # Previous value of u
Iyy = 3.45

# Define the differential equations
def remus_vehicle(t, x, u_input, params):
    global i
    i = int(t * 1000)  # Convert time to index
    u, w, q, z, theta = x
    delta_s = u_input[0]  # External control input
    X_HS = -(W - B) * np.sin(theta)
    Z_HS = (W - B)*np.cos(theta)
    M_HS = -(zg * W - zb * B) * np.sin(theta) - (xg * W - xb * B) * np.cos(theta)

    # coeff = np.array([
    #     [m - X_udot, 0, m * zg, 0, 0],
    #     [0, m - Z_wdot, -(m * xg + Z_qdot), 0, 0],
    #     [m * zg, -(m * xg + M_wdot), Iyy - M_qdot, 0, 0],
    #     [0, 0, 0, 1, 0],
    #     [0, 0, 0, 0, 1]
    # ])
    coeff = np.array([
        [(m-X_udot), 0, m*zg],
        [0, (m-Z_wdot), -(m*xg+Z_qdot)],
        [m*zg, -(m*xg+M_wdot), (Iyy-M_qdot)]
        ])
    coeff_inv = np.linalg.inv(coeff)
    # summation = np.array([
    #     [X_HS + X_uu * u * abs(u) + (X_wq - m) * w * q + (X_qq + m * xg) * q**2 + X_prop],
    #     [Z_HS + Z_ww * w * abs(w) + Z_qq * q * abs(q) + (Z_uq + m) * u * q + Z_uw * u * w + m * zg * (q**2) + Z_uu_delta_s * u**2 * delta_s],
    #     [M_HS + M_ww * w * abs(w) + M_qq * q * abs(q) + (M_uq - m * xg) * u * q - m * zg * w * q + M_uw * u * w + M_uu_delta_s * u**2 * delta_s],
    #     [-u*np.sin(theta) + w * np.cos(theta)],
    #     [q]
    # ])
    summation = np.array([
        [X_HS + X_uu * u * abs(u) + (X_wq - m) * w * q + ((X_qq + m * xg) * q**2) + X_prop],
        [Z_HS + Z_ww * w * abs(w) + Z_qq * q * abs(q) + (Z_uq + m) * u * q + Z_uw * u * w + Z_uu_delta_s * u**2 * delta_s],
        [M_HS + M_ww * w * abs(w) + M_qq * q * abs(q) + (M_uq - m * xg) * u * q - m * zg * w * q + M_uw * u * w + M_uu_delta_s * u**2 * delta_s],
     ])
    z_dot= -u*np.sin(theta)+w*np.cos(theta)
    theta_dot=q
    dot = np.matmul(coeff_inv, summation)
    #return np.array([dot[0][0], dot[1][0], dot[2][0], dot[3][0], dot[4][0]])
    return np.array([dot[0][0], dot[1][0], dot[2][0],z_dot,theta_dot])

# Initial conditions
initial_state = [1.5, 0.0, 0.0, 0.0, 0.0]

# Time array
t = np.arange(0, 50, 0.001)

# External control input array (matching the pattern in the image)
u_input = np.zeros_like(t)
# u_input[1000:4000] = 0.04  # 1 to 4 seconds
# u_input[4000:5000] = -0.04  # 4 to 5 seconds
# u_input[5000:7000] = 0.02  # 5 to 7 seconds
# u_input[7000:9000] = -0.02  # 7 to 9 seconds
# u_input[9000:] = 0.0  # 9 to 20 seconds
u_input[:1000] = -10*np.pi/180  # 0 to 1 seconds
u_input[1000:3000] = 10*np.pi/180  # 1 to 2 seconds
u_input[3000:5000] = -10*np.pi/180 # 2 to 3 seconds
u_input[5000:7000] = 10*np.pi/180  # 3 to 4 seconds
u_input[7000:9000] =-10*np.pi/180
u_input[9000:12000]= 0.0


u_input[9000:] = 0.0  # 9 to 20 seconds
u_input[:-1]=0.0


# Define the filter transfer function
numerator = [0.5]
denominator = [1, 0.5]
filter_tf = ctrl.TransferFunction(numerator, denominator)

# Apply the filter to the input
t_out, u_filtered = ctrl.forced_response(filter_tf, T=t, U=u_input)

# Create NonlinearIOSystem
nonlinear_system = ctrl.NonlinearIOSystem(
    updfcn=remus_vehicle,  # State derivative function
    inputs=1, outputs=5, states=5,
    name='remus_vehicle'
)

# Simulate the system with filtered input
T, yout = ctrl.input_output_response(nonlinear_system, T=t, U=u_filtered, X0=initial_state)

# Plot the simulation output
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
plt.plot(T, u_filtered, label='Filtered stern angle')
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
