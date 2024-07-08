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
u_true= yout[0]
w_true= yout[1]
q_true= yout[2]
z_true= yout[3]
theta_true= yout[4]
t=T

t_tensor = torch.tensor(t, dtype=torch.float32).view(-1, 1)
u_true_tensor = torch.tensor(u_true, dtype=torch.float32).view(-1, 1)
w_true_tensor = torch.tensor(w_true, dtype=torch.float32).view(-1, 1)
q_true_tensor = torch.tensor(q_true, dtype=torch.float32).view(-1, 1)
z_true_tensor = torch.tensor(z_true, dtype=torch.float32).view(-1, 1)
theta_true_tensor = torch.tensor(theta_true, dtype=torch.float32).view(-1, 1)

# Define the PINN model
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.fc1 = nn.Linear(1, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 100)
        self.fc4 = nn.Linear(100, 100)
        # self.fc5 = nn.Linear(50, 50)
        # self.fc6 = nn.Linear(50, 50)
        # self.fc7 = nn.Linear(50, 50)
        # self.fc8 = nn.Linear(50, 50)
        # self.fc9 = nn.Linear(50, 50)
        # self.fc10 = nn.Linear(50, 50)
        self.out_u = nn.Linear(100, 1)  # Output for u (x-direction velocity)
        self.out_w = nn.Linear(100, 1)  # Output for w (z-direction velocity)
        self.out_q = nn.Linear(100, 1)  # Output for q (pitch rate)
        self.out_z = nn.Linear(100, 1)  # Output for z (z direction position)
        self.out_theta = nn.Linear(100, 1)  # Output for theta (pitch angle)
        
        # Randomly initialize learnable parameter within a reasonable range
        self.X_uu = nn.Parameter(torch.tensor([np.random.uniform(-5.0, 5.0)], dtype=torch.float32,requires_grad=True))  # Initial guess for X_uu
        self.X_wq = nn.Parameter(torch.tensor([np.random.uniform(-5.0, 5.0)], dtype=torch.float32,requires_grad=True))  # Initial guess for X_wq
        self.X_qq = nn.Parameter(torch.tensor([np.random.uniform(-5.0, 5.0)], dtype=torch.float32,requires_grad=True))  # Initial guess for X_qq
        self.Z_ww = nn.Parameter(torch.tensor([np.random.uniform(-5.0, 5.0)], dtype=torch.float32,requires_grad=True))  # Initial guess for Z_ww
        self.Z_qq = nn.Parameter(torch.tensor([np.random.uniform(-5.0, 5.0)], dtype=torch.float32,requires_grad=True))  # Initial guess for Z_qq
        self.Z_uq = nn.Parameter(torch.tensor([np.random.uniform(-5.0, 5.0)], dtype=torch.float32,requires_grad=True))  # Initial guess for Z_uq
        self.Z_uw = nn.Parameter(torch.tensor([np.random.uniform(-5.0, 5.0)], dtype=torch.float32,requires_grad=True))  # Initial guess for Z_uw
        self.Z_uu_delta_s = nn.Parameter(torch.tensor([np.random.uniform(-5.0, 5.0)], dtype=torch.float32,requires_grad=True))  # Initial guess for _uu_delta_s
        self.M_ww = nn.Parameter(torch.tensor([np.random.uniform(-5.0, 5.0)], dtype=torch.float32,requires_grad=True))  # Initial guess for M_ww
        self.M_qq = nn.Parameter(torch.tensor([np.random.uniform(-5.0, 5.0)], dtype=torch.float32,requires_grad=True))  # Initial guess for M_qq
        self.M_uq = nn.Parameter(torch.tensor([np.random.uniform(-5.0, 5.0)], dtype=torch.float32,requires_grad=True))  # Initial guess for M_uq
        self.M_uw = nn.Parameter(torch.tensor([np.random.uniform(-5.0, 5.0)], dtype=torch.float32,requires_grad=True))  # Initial guess for M_uw
        self.M_uu_delta_s = nn.Parameter(torch.tensor([np.random.uniform(-5.0, 5.0)], dtype=torch.float32))  # Initial guess for M_uu_delta_s
    
    def forward(self, t):
        # x = torch.tanh(self.fc1(t))
        # x = torch.tanh(self.fc2(x))
        # x = torch.tanh(self.fc3(x))
        # x = torch.tanh(self.fc4(x))
        # x = torch.tanh(self.fc5(x))
        # x = torch.tanh(self.fc6(x))
        # x = torch.tanh(self.fc7(x))
        # x = torch.tanh(self.fc8(x))
        # x = torch.tanh(self.fc9(x))
        # x = torch.tanh(self.fc10(x))
        x = nn.GELU()(self.fc1(t))
        x =nn.GELU()(self.fc2(x))
        x = nn.GELU()(self.fc3(x))
        x = nn.GELU()(self.fc4(x))
        # x = nn.GELU()(self.fc5(x))
        # x = nn.GELU()(self.fc6(x))
        # x = nn.GELU()(self.fc7(x))
        # x = nn.GELU()(self.fc8(x))
        # x = nn.GELU()(self.fc9(x))
        # x = nn.GELU()(self.fc10(x))
        u = self.out_u(x)
        w = self.out_w(x)
        q = self.out_q(x)
        z = self.out_z(x)
        theta = self.out_theta(x)
        return u,w,q,z,theta

    def physics_model(self, t, u_input):
        u, w, q, z, theta = self.forward(t)
        delta_s=u_input[0]
        X_HS = -(W - B) * torch.sin(theta)
        Z_HS = (W - B)*torch.cos(theta)
        M_HS = -(zg * W - zb * B) * torch.sin(theta) - (xg * W - xb * B) * torch.cos(theta)
        coeff = torch.tensor([
        [(m-X_udot), 0, m*zg],
        [0, (m-Z_wdot), -(m*xg+Z_qdot)],
        [m*zg, -(m*xg+M_wdot), (Iyy-M_qdot)]
        ])
        coeff_inv=torch.inverse(coeff)
        X_uu = self.X_uu
        X_wq = self.X_wq
        X_qq = self.X_qq
        Z_ww = self.Z_ww
        Z_qq = self.Z_qq
        Z_uq = self.Z_uq
        Z_uw = self.Z_uw
        Z_uu_delta_s = self.Z_uu_delta_s
        M_ww = self.M_ww
        M_qq = self.M_qq
        M_uq = self.M_uq
        M_uw = self.M_uw
        M_uu_delta_s = model.M_uu_delta_s
        summation = torch.stack([
        X_HS + X_uu * u * torch.abs(u) + (X_wq - m) * w * q + (X_qq + m * xg) * q**2 + X_prop,
        Z_HS + Z_ww * w * torch.abs(w) + Z_qq * q * torch.abs(q) + (Z_uq + m) * u * q + Z_uw * u * w  + Z_uu_delta_s * u**2 * delta_s,
        M_HS + M_ww * w * torch.abs(w) + M_qq * q * torch.abs(q) + (M_uq - m * xg) * u * q - m * zg * w * q + M_uw * u * w + M_uu_delta_s * u**2 * delta_s,
        ], dim=0)
        coeff_inv = coeff_inv.repeat(1, t.shape[0], 1)
        dot = torch.bmm(coeff_inv, summation.permute(2, 0, 1)).permute(2, 0, 1)
        z_dot=-u*torch.sin(theta)+w*torch.cos(theta)
        theta_dot=q
        u_dot =dot[:, :, 0].squeeze(0)
        w_dot = dot[:, :, 10000].squeeze(0)
        q_dot = dot[:, :, 20000].squeeze(0) 

        return u_dot,w_dot,q_dot,z_dot,theta_dot
        
# Instantiate the model
model = PINN()


def physics_loss(model, t,u_true_tensor,w_true_tensor,q_true_tensor,z_true_tensor,theta_true_tensor):
    t = t.requires_grad_(True)
    u, w, q, z, theta = model(t)
    
    #Compute derivatives with respect to time
    du_dt = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    dw_dt = torch.autograd.grad(w, t, grad_outputs=torch.ones_like(w), create_graph=True)[0]
    dq_dt = torch.autograd.grad(q, t, grad_outputs=torch.ones_like(q), create_graph=True)[0]
    dz_dt = torch.autograd.grad(z, t, grad_outputs=torch.ones_like(z), create_graph=True)[0]
    dtheta_dt = torch.autograd.grad(theta, t, grad_outputs=torch.ones_like(theta), create_graph=True)[0]

    yout_torch=model.physics_model(t,torch.tensor(u_input))

    physics_residual_u = du_dt - yout_torch[0]
    physics_residual_w = dw_dt - yout_torch[1]
    physics_residual_q = dq_dt - yout_torch[2]
    physics_residual_z = dz_dt - yout_torch[3]
    physics_residual_theta= dtheta_dt-yout_torch[4]
    
    # Compute mean squared error of the residuals
    return torch.mean(physics_residual_u ** 2 + physics_residual_w ** 2 + physics_residual_q ** 2 + physics_residual_z ** 2 + physics_residual_theta**2 )
    
def data_loss(model, t,u_true,w_true,q_true,z_true,theta_true):
    u_pred,w_pred,q_pred,z_pred,theta_pred = model(t)
    return torch.mean((u_pred - u_true)**2)+torch.mean((w_pred - w_true)**2)+torch.mean((q_pred - q_true)**2)+torch.mean((z_pred - z_true)**2)+ torch.mean((theta_pred - theta_true)**2)

def initial_condition_loss(model, t0,u0,w0,q0,z0,theta0):
    u_pred,w_pred,q_pred,z_pred,theta_pred = model(t0)
    return torch.mean((u_pred - u0)**2+(w_pred - w0)**2+(q_pred - q0)**2+(z_pred - z0)**2+ (theta_pred - theta0)**2)

# Initial condition tensors
t0_tensor = torch.tensor([[0]], dtype=torch.float32)
u0_tensor = torch.tensor([[u0]], dtype=torch.float32)
w0_tensor = torch.tensor([[w0]], dtype=torch.float32)
q0_tensor = torch.tensor([[q0]], dtype=torch.float32)
z0_tensor = torch.tensor([[z0]], dtype=torch.float32)
theta0_tensor = torch.tensor([[theta0]], dtype=torch.float32)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-2)
#optimizer = torch.optim.LBFGS(model.parameters(), lr=0.001, max_iter=1000, max_eval=None, tolerance_grad=1e-11, tolerance_change=1e-11, history_size=100, line_search_fn='strong_wolfe')


# Training loop
epochs = 10000
for epoch in range(epochs):
    optimizer.zero_grad()
    loss = (physics_loss(model, t_tensor,u_true_tensor,w_true_tensor,q_true_tensor,z_true_tensor,theta_true_tensor) +
            data_loss(model, t_tensor, u_true_tensor,w_true_tensor,q_true_tensor,z_true_tensor,theta_true_tensor) +
            initial_condition_loss(model, t0_tensor,u0_tensor,w0_tensor,q0_tensor,z0_tensor,theta0_tensor))
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}, X_uu: {model.X_uu.item()},X_wq: {model.X_wq.item()}")


# Predicting using the trained model
with torch.no_grad():
    u_pred,w_pred,q_pred,z_pred,theta_pred = model(t_tensor)
    u_pred_pred = u_pred.numpy().flatten()
    w_pred_pred = w_pred.numpy().flatten()
    q_pred = q_pred.numpy().flatten()
    z_pred = z_pred.numpy().flatten()
    theta_pred = theta_pred.numpy().flatten()


# Print the learned parameter
print(f"Learned X_uu: {model.X_uu.item()}")
print(f"Learned X_wq: {model.X_wq.item()}")
print(f"Learned X_qq: {model.X_qq.item()}")
print(f"Learned Z_ww: {model.Z_ww.item()}")
print(f"Learned Z_qq: {model.Z_qq.item()}")
print(f"Learned Z_uq: {model.Z_uq.item()}")
print(f"Learned Z_uw: {model.Z_uw.item()}")
print(f"Learned Z_uu_delta_s: {model.Z_uu_delta_s.item()}")
print(f"Learned M_ww: {model.M_ww.item()}")
print(f"Learned M_qq: {model.X_uu.item()}")
print(f"Learned M_uq: {model.M_uq.item()}")
print(f"Learned M_uw: {model.M_uw.item()}")
print(f"Learned M_uu_delta_s: {model.M_uu_delta_s.item()}")

# Plotting the results
plt.figure(figsize=(10, 5))


plt.subplot(5, 1, 1)
plt.plot(t, u_true, 'b-', label='True u')
plt.plot(t, u_pred, 'r--', label='Predicted u')
plt.xlabel('Time (s)')
plt.ylabel('u (m/s)')
plt.title('u pred v/s true')
plt.legend()
plt.subplot(5, 1, 2)
plt.plot(t, w_true, 'b-', label='True u')
plt.plot(t, w_pred, 'r--', label='Predicted u')
plt.xlabel('Time (s)')
plt.ylabel('w (m/s)')
plt.title('w pred v/s true')
plt.legend()
plt.subplot(5, 1, 3)
plt.plot(t, q_true, 'b-', label='True u')
plt.plot(t, q_pred, 'r--', label='Predicted u')
plt.xlabel('Time (s)')
plt.ylabel('q (rad/s)')
plt.title('Pitch rate q pred v/s true')
plt.legend()
plt.subplot(5, 1, 4)
plt.plot(t, z_true, 'b-', label='True u')
plt.plot(t, z_pred, 'r--', label='Predicted u')
plt.xlabel('Time (s)')
plt.ylabel('z (m)')
plt.title('z pred v/s true')
plt.legend()
plt.subplot(5, 1, 5)
plt.plot(t, theta_true, 'b-', label='True u')
plt.plot(t, theta_pred, 'r--', label='Predicted u')
plt.xlabel('Time (s)')
plt.ylabel('theta (rad)')
plt.title('theta pred v/s true')
plt.legend()
plt.show()
