import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

# Load pH Data from Excel file
pH_data = pd.read_excel(r"Exp pH values.xlsx")
times_min_raw = pH_data.iloc[:, 0]
times_min = pd.to_numeric(times_min_raw, errors='coerce')
valid_indices = ~times_min.isna()
times_min_clean = times_min[valid_indices].values
pH_values_all_clean = pH_data.loc[valid_indices, pH_data.columns[1:9]].values
times_sec = times_min_clean * 60  # convert minutes to seconds

# Add central cavity zone 0 pH with constant pH
cavity_pH_value = 3.7
cavity_pH_array = np.full((pH_values_all_clean.shape[0], 1), cavity_pH_value)
pH_values_extended = np.hstack((cavity_pH_array, pH_values_all_clean))
num_radial_zones = pH_values_extended.shape[1]

# Enzyme activity factor
def enzyme_activity_factor(pH):
    pH_opt = 7.0
    sigma = 1.5
    return np.exp(-((pH - pH_opt) ** 2) / (2 * sigma ** 2))

# Interpolators for each zone including cavity
pH_interpolators = []
for i in range(num_radial_zones):
    pH_zone = pH_values_extended[:, i]
    interp_func = interp1d(times_sec, pH_zone, bounds_error=False, fill_value="extrapolate")
    pH_interpolators.append(interp_func)

# Model parameters
D_A = 5e-11 * 1e11
D_I = 2e-11 * 1e11
D_B = 4e-10 * 1e11
D_P = 2e-11 * 1e11

kcat = 1e4
km = 3.3e-6
kf = 1.0
kb = 1e3

L = 8.0
T = 0.1
N_t = 10
N_r, N_theta = num_radial_zones, 21
dr = L / N_r
dtheta = 2 * np.pi / N_theta
r = np.linspace(0, L - dr / 2, N_r)
theta = np.linspace(0, 2 * np.pi, N_theta)
R, Theta = np.meshgrid(r, theta, indexing='ij')
X, Y = R * np.cos(Theta), R * np.sin(Theta)

# Initial condition
shape_2d = (N_r, N_theta)
C_A = np.zeros(shape_2d)
C_I = np.zeros(shape_2d)
C_B = np.zeros(shape_2d)
C_E = 15e-9 * np.ones(shape_2d)
C_R = 0.5e-3 * np.ones(shape_2d)
C_P = np.zeros(shape_2d)

C_B[0, :] = 50.0 * 10**(-3)   # Inject ProA in the center
C_I[0, :] = 5.0 * 10**(-3)    # Inject Inhibitor in the center

# Time scaling
time_scale = 90000  # 150 min * 60 s/min / 0.1 model sec

# Reaction-Diffusion system
def reaction_diffusion(t, C_flat):
    C = C_flat.reshape((6, N_r, N_theta))
    C_A, C_B, C_I, C_E, C_R, C_P = C
    dC_A, dC_B, dC_I, dC_E, dC_R, dC_P = (np.zeros_like(C_A) for _ in range(6))
    for i in range(N_r):
        # Scaling model time to real time before interpolating pH
        local_pH = pH_interpolators[i](t * time_scale) if i > 0 else 0
        activity_factor = enzyme_activity_factor(local_pH) if i > 0 else 0.0
        for j in range(N_theta):
            jp = (j + 1) % N_theta
            jm = (j - 1) % N_theta

            def diffusion_term(C_species):
                if i == 0:
                    d2C_dr2 = (C_species[1, j] - C_species[0, j]) / dr**2
                    dC_dr = 0
                elif i == N_r - 1:
                    d2C_dr2 = (C_species[i, j] - C_species[i - 1, j]) / dr**2
                    dC_dr = 0
                else:
                    d2C_dr2 = (C_species[i+1, j] - 2*C_species[i, j] + C_species[i-1, j]) / dr**2
                    dC_dr = (C_species[i+1, j] - C_species[i-1, j]) / (2*dr)
                d2C_dtheta2 = (C_species[i, jp] - 2*C_species[i, j] + C_species[i, jm]) / dtheta**2
                return d2C_dr2, dC_dr, d2C_dtheta2

            d2C_dr2, dC_dr, d2C_dtheta2 = diffusion_term(C_A)
            dC_A[i, j] = D_A * (d2C_dr2 + (1/(r[i]+1e-10)) * dC_dr + (1/(r[i]+1e-10)**2) * d2C_dtheta2)

            d2C_dr2, dC_dr, d2C_dtheta2 = diffusion_term(C_B)
            dC_B[i, j] = D_B * (d2C_dr2 + (1/(r[i]+1e-10)) * dC_dr + (1/(r[i]+1e-10)**2) * d2C_dtheta2)

            d2C_dr2, dC_dr, d2C_dtheta2 = diffusion_term(C_I)
            dC_I[i, j] = D_I * (d2C_dr2 + (1/(r[i]+1e-10)) * dC_dr + (1/(r[i]+1e-10)**2) * d2C_dtheta2)

            d2C_dr2, dC_dr, d2C_dtheta2 = diffusion_term(C_P)
            dC_P[i, j] = D_P * (d2C_dr2 + (1/(r[i]+1e-10)) * dC_dr + (1/(r[i]+1e-10)**2) * d2C_dtheta2)

            # Reaction kinetics
            reaction_AB = (kcat * activity_factor * C_B[i, j] * C_E[i, j]) / (km + C_B[i, j]) if i > 0 else 0
            reaction_RP = kf * (C_R[i, j] * np.exp(-kf*C_A[i,j] - kb*C_I[i,j]) * t) * C_A[i, j]

            dC_A[i, j] += reaction_AB
            dC_B[i, j] -= reaction_AB
            dC_P[i, j] += reaction_RP

    return np.ravel([dC_A, dC_B, dC_I, dC_E, dC_R, dC_P])

t_eval = np.linspace(0, T, N_t)
C0 = np.ravel([C_A, C_B, C_I, C_E, C_R, C_P])
sol = solve_ivp(reaction_diffusion, [0, T], C0, t_eval=t_eval, method='RK45', dense_output=False)
C_A_sol, C_B_sol, C_I_sol, C_E_sol, C_R_sol, C_P_sol = sol.y.reshape((6, N_r, N_theta, -1))

# pH plot for all zones
plt.figure()
for i in range(num_radial_zones):
    plt.plot(times_sec, [pH_interpolators[i](t) for t in times_sec], label=f'pH zone {i}')
plt.xlabel('Time (s)')
plt.ylabel('pH')
plt.legend()
plt.title('pH evolution at radial zones including central cavity')
plt.show()

# Plot of enzyme activity vs pH
pH_range = np.linspace(0, 14, 100)
activity_curve = enzyme_activity_factor(pH_range)
plt.figure()
plt.plot(pH_range, activity_curve)
plt.xlabel('pH')
plt.ylabel('Enzyme Activity Factor')
plt.title('Enzyme Activity vs. pH')
plt.show()
