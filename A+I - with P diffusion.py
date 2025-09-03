import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define parameters
D_A = 2*10**(-11) * 10**(11)
D_I = 2*10**(-11) * 10**(11)
D_P = 2*10**(-11) * 10**(11)

kf = 1.0
kb = 10e3

L = 8.0
T = 0.1
N_t = 10
N_r, N_theta = 9, 21  
dr = L / N_r
dtheta = 2 * np.pi / N_theta
r = np.linspace(0, L - dr / 2, N_r)
theta = np.linspace(0, 2 * np.pi, N_theta)
R, Theta = np.meshgrid(r, theta, indexing='ij')
X, Y = R * np.cos(Theta), R * np.sin(Theta)

# Initial conditions
C_A = np.zeros((N_r, N_theta))
C_I = np.zeros((N_r, N_theta))
C_R = 0.5 * 10**(-3) * np.ones((N_r, N_theta))
C_P = np.zeros((N_r, N_theta))

C_A[0, :] = 50.0 * 10**(-3)   # Inject A in the center
C_I[0, :] = 5.0 * 10**(-3)    # Inject Inhibitor in the center

# Reaction-diffusion equations
def reaction_diffusion(t, C_flat):
    C = C_flat.reshape((4, N_r, N_theta))
    C_A, C_I, C_R, C_P = C
    dC_A, dC_I, dC_R, dC_P = np.zeros_like(C_A), np.zeros_like(C_I), np.zeros_like(C_R), np.zeros_like(C_P)
    
    for i in range(N_r):
        for j in range(N_theta):
            jp = (j + 1) % N_theta
            jm = (j - 1) % N_theta
            
            if i == 0:
                # Neumann BC at r=0 (zero flux)
                d2C_dr2 = (C_A[1, j] - C_A[0, j]) / dr**2
                dC_dr = 0
            elif i == N_r - 1:
                # Neumann BC at r=L (zero flux)
                d2C_dr2 = (C_A[i, j] - C_A[i - 1, j]) / dr**2
                dC_dr = 0
            else:
                d2C_dr2 = (C_A[i + 1, j] - 2 * C_A[i, j] + C_A[i - 1, j]) / dr**2
                dC_dr = (C_A[i + 1, j] - C_A[i - 1, j]) / (2 * dr)

            d2C_dtheta2 = (C_A[i, jp] - 2 * C_A[i, j] + C_A[i, jm]) / dtheta**2
            dC_A[i, j] = D_A * (d2C_dr2 + (1 / (r[i] + 1e-10)) * dC_dr + (1 / (r[i] + 1e-10)**2) * d2C_dtheta2)
            
            
            if i == 0:
                d2C_dr2 = (C_I[1, j] - C_I[0, j]) / dr**2
                dC_dr = 0
            elif i == N_r - 1:
                d2C_dr2 = (C_I[i, j] - C_I[i - 1, j]) / dr**2
                dC_dr = 0
            else:
                d2C_dr2 = (C_I[i + 1, j] - 2 * C_I[i, j] + C_I[i - 1, j]) / dr**2
                dC_dr = (C_I[i + 1, j] - C_I[i - 1, j]) / (2 * dr)
            
            d2C_dtheta2 = (C_I[i, jp] - 2 * C_I[i, j] + C_I[i, jm]) / dtheta**2
            dC_I[i, j] = D_I * (d2C_dr2 + (1 / (r[i] + 1e-10)) * dC_dr + (1 / (r[i] + 1e-10)**2) * d2C_dtheta2)


            # Diffusion of P
            if i == 0:
                d2C_dr2 = (C_P[1, j] - C_P[0, j]) / dr**2
                dC_dr = 0
            elif i == N_r - 1:
                d2C_dr2 = (C_P[i, j] - C_P[i - 1, j]) / dr**2
                dC_dr = 0
            else:
                d2C_dr2 = (C_P[i + 1, j] - 2*C_P[i, j] + C_P[i - 1, j]) / dr**2
                dC_dr = (C_P[i + 1, j] - C_P[i - 1, j]) / (2*dr)
            d2C_dtheta2 = (C_P[i, jp] - 2*C_P[i, j] + C_P[i, jm]) / dtheta**2
            dC_P[i, j] += D_P * (d2C_dr2 + (1/(r[i] + 1e-10)) * dC_dr + (1/(r[i] + 1e-10)**2) * d2C_dtheta2)
            
            reaction_RP = kf * (C_R[i, j] * np.exp(- kf * C_A[i, j] - kb * C_I[i, j]) * t) * C_A[i, j]
            dC_P[i, j] += reaction_RP
    
    return np.ravel([dC_A, dC_I, dC_R, dC_P])

t_eval = np.linspace(0, T, N_t)
C0 = np.ravel([C_A, C_I, C_R, C_P])
sol = solve_ivp(reaction_diffusion, [0, 1], C0, t_eval=t_eval, method='RK45')
C_A_sol, C_I_sol, C_R_sol, C_P_sol = sol.y.reshape((4, N_r, N_theta, -1))

# print(sol.success)  # Should be True if solver finished successfully
# print(sol.message)  # May show if it stopped early or faced errors
