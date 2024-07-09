# AIRL_UAV_Programs

#Orginal/True parameter values:
params = {
    'X_uu': -1.62,
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

#Necessary equations of the model:
Coefficient matrix (constant):
coeff = ([
    [(m-X_udot), 0, m*zg],
    [0, (m-Z_wdot), -(m*xg+Z_qdot)],
    [m*zg, -(m*xg+M_wdot), (Iyy-M_qdot)]
    ])

Summation matrix:
summation = ([
    [X_HS + X_uu * u*abs(u) + (X_wq - m) * w * q + (X_qq + m * xg) * q**2 + X_prop],
    [Z_HS + Z_ww * w*abs(w) + Z_qq * q*abs(q) + (Z_uq + m) * u * q + Z_uw * u * w + Z_uu_delta_s * u**2 * delta_s],
    [M_HS + M_ww * w*abs(w) + M_qq * q*abs(q) + (M_uq - m * xg) * u * q + M_uw * u * w + M_uu_delta_s * u**2 * delta_s]
    ])

[u_dot,w_dot,q_dot]=coeff_inverse*summation

Separate compuatations:
z_dot=-u*np.sin(theta)+w*np.cos(theta)
theta_dot=q

**************SINDY**************************************************
feature/candidate library for SINDY:
functions = [1,u,w,q,z,theta,u|u|,w|w|,q|q|,q**2,wq,uq,uw,sin(theta),cos(theta),u^2*delta_s]


