import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Diffusion coefficients
D_A = 2*10**(-11) * 10**(11)   # Activator
D_I = 2*10**(-11) * 10**(11)   # Inhibitor
D_B = 4*10**(-10) * 10**(11)   # ProA
D_P = 2*10**(-11) * 10**(11)   # Product

# Reaction parameters
kcat = 10**(4)
km = 3.3 * 10**(-6)
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
C_A = np.zeros((N_r, N_theta))   # Concentration of Activator
C_I = np.zeros((N_r, N_theta))   # Concentration of Inhibitor
C_B = np.zeros((N_r, N_theta))   # Concentration of ProA
C_E = 15.0 * 10**(-9) * np.ones((N_r, N_theta))   # Concentration of Enzyme
C_R = 0.5 * 10**(-3) * np.ones((N_r, N_theta))   # Concentration of Reactant
C_P = np.zeros((N_r, N_theta))   # Concentration of Product

C_B[0, :] = 50.0 * 10**(-3)   # Inject ProA in the centre
C_I[0, :] = 5.0 * 10**(-3)    # Inject Inhibitor in the centre

# Reaction-diffusion system
def reaction_diffusion(t, C_flat):
    C = C_flat.reshape((6, N_r, N_theta))
    C_A, C_B, C_I, C_E, C_R, C_P = C
    dC_A, dC_B, dC_I, dC_E, dC_R, dC_P = np.zeros_like(C_A), np.zeros_like(C_B), np.zeros_like(C_I), np.zeros_like(C_E), np.zeros_like(C_R), np.zeros_like(C_P)
    
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
                d2C_dr2 = (C_B[1, j] - C_B[0, j]) / dr**2
                dC_dr = 0
            elif i == N_r - 1:
                d2C_dr2 = (C_B[i, j] - C_B[i - 1, j]) / dr**2
                dC_dr = 0
            else:
                d2C_dr2 = (C_B[i + 1, j] - 2 * C_B[i, j] + C_B[i - 1, j]) / dr**2
                dC_dr = (C_B[i + 1, j] - C_B[i - 1, j]) / (2 * dr)
            
            d2C_dtheta2 = (C_B[i, jp] - 2 * C_B[i, j] + C_B[i, jm]) / dtheta**2
            dC_B[i, j] = D_B * (d2C_dr2 + (1 / (r[i] + 1e-10)) * dC_dr + (1 / (r[i] + 1e-10)**2) * d2C_dtheta2)
            
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

            if i == 0:                 # Diffusion of P
                d2C_dr2 = (C_P[1, j] - C_P[0, j]) / dr**2
                dC_dr = 0
            elif i == N_r - 1:
                d2C_dr2 = (C_P[i, j] - C_P[i - 1, j]) / dr**2
                dC_dr = 0
            else:
                d2C_dr2 = (C_P[i + 1, j] - 2 * C_P[i, j] + C_P[i - 1, j]) / dr**2
                dC_dr = (C_P[i + 1, j] - C_P[i - 1, j]) / (2 * dr)
            
            d2C_dtheta2 = (C_P[i, jp] - 2 * C_P[i, j] + C_P[i, jm]) / dtheta**2
            
            dC_P[i, j] += D_P * (d2C_dr2 + (1 / (r[i] + 1e-10)) * dC_dr + (1 / (r[i] + 1e-10)**2) * d2C_dtheta2)
            
            reaction_AB = (kcat * C_B[i, j] * C_E[i, j]) / (km + C_B[i, j])
            reaction_RP = kf * (C_R[i, j] * np.exp(- kf * C_A[i, j] - kb * C_I[i, j]) * t) * C_A[i, j]

            dC_A[i, j] += reaction_AB
            dC_B[i, j] -= reaction_AB
            dC_P[i, j] += reaction_RP
    
    return np.ravel([dC_A, dC_B, dC_I, dC_E, dC_R, dC_P])

t_eval = np.linspace(0, T, N_t)
C0 = np.ravel([C_A, C_B, C_I, C_E, C_R, C_P])
sol = solve_ivp(reaction_diffusion, [0, T], C0, t_eval=t_eval, method='RK45', dense_output=False)
C_A_sol, C_B_sol, C_I_sol, C_E_sol, C_R_sol, C_P_sol = sol.y.reshape((6, N_r, N_theta, -1))

# print(sol.success)  # Should be True if solver finished successfully
# print(sol.message)  # May show if it stopped early or faced errors
