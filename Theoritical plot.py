# Run this code right after ProA+I.py or A+I.py

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# %matplotlib notebook

# Visualization - 3D Surface Plots
species = {'P': C_P_sol}

rows, cols = 4, 3  # Grid layout per species

for name, C_sol in species.items():
    num_plots = len(t_eval)
    fig, axes = plt.subplots(rows, cols, figsize=(12, 16), subplot_kw={'projection': '3d'})
    fig.suptitle(f'Concentration Profiles of {name}, with Pro A at Cb = {(C_B[0][0])} and Ci = {(C_I[0][0])}', fontsize=16)
    
    # Global min and max for color scaling - normalised
    vmin, vmax = 0, 2.5e-11
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap('turbo')
    
    surf_handles = []
    for i, t in enumerate(t_eval):
        row, col = divmod(i, cols)
        ax = axes[row, col]
        surf = ax.plot_surface(X, Y, C_sol[:, :, i], cmap='turbo', norm=norm, edgecolor='w', linewidth=0.3)
        surf_handles.append(surf)
        
        # Rotate the plot for a top-down view
        ax.view_init(elev=90, azim=-90)
        
        ax.set_title(f't={t:.3f}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zticks([])

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.05, pad=0.04, label=f'Concentration of {name} (mM)', location='left')
    
    # Hide any unused subplots
    for i in range(num_plots, rows * cols):
        row, col = divmod(i, cols)
        fig.delaxes(axes[row, col])
    
    plt.show()