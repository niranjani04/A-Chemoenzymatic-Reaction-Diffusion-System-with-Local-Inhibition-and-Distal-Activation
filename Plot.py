# import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors 

species = {'P': C_P_sol}

rows, cols = 4, 3

for name, C_sol in species.items():
    num_plots = len(t_eval)
    fig, axes = plt.subplots(rows, cols, figsize=(12, 16), subplot_kw={'projection': '3d'})
    fig.suptitle(f'[{name}], with Pro A at ProA = {C_B[0, :][0]*1e3} mM and I = {C_I[0, :][0]*1e3} mM, kcat = {kcat} and kb= {kb}, D_B = {D_B*10**(-11)*10**(11):.0f}, D_IN = {D_I*10**(-11)*10**(11):.0f}, D_A = {D_A*10**(-10)*10**(10):.0f}', fontsize=16)
    # fig.suptitle(f'[{name}], without Pro A at A = {C_A[0, :][0]*1e3} mM and I = {C_I[0, :][0]*1e3} mM, kb= {kb}, D_IN = {D_I*10**(-11)*10**(11):.0f}, D_A = {D_A*10**(-10)*10**(10):.0f}', fontsize=16)

    vmin, vmax = 0, C_sol.max()
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
        ax.set_zticks([])  # Hide z-axis ticks for top view

    # Add one colorbar for the entire figure
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.05, pad=0.04, label=f'[{name}]', location='left')
    
    # Hide any unused subplots
    for i in range(num_plots, rows * cols):
        row, col = divmod(i, cols)
        fig.delaxes(axes[row, col])
    
    plt.show()
