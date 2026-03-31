import numpy as np
import matplotlib.pyplot as plt

def generate_theoretical_plots():
  
    plt.rcParams.update({
        'font.size': 14,
        'font.family': 'serif',
        'axes.labelsize': 18,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'figure.autolayout': True
    })


    pos = np.linspace(-20, 5, 500)
    vel = np.linspace(0, 15, 500)
    P, V = np.meshgrid(pos, vel)


    max_decel = 3.0


    h_orig = -P 


    h_inner = -P - (V**2) / (2 * max_decel)


    fig1, ax1 = plt.subplots(figsize=(8, 6))
    
   
    ax1.contourf(P, V, h_orig, levels=[0, np.inf], colors=['#A8DADC'], alpha=0.7)

    ax1.contour(P, V, h_orig, levels=[0], colors=['#1D3557'], linewidths=3)
    
    ax1.set_xlim(-20, 5)
    ax1.set_ylim(0, 15)
    ax1.set_xlabel('Position $p$ [m]')
    ax1.set_ylabel('Velocity $v$ [m/s]')
 
    import matplotlib.patches as mpatches
    safe_patch = mpatches.Patch(color='#A8DADC', label='Original Safe Set $\mathcal{C}$ ($h(x) \geq 0$)')
    ax1.legend(handles=[safe_patch], loc='upper left')
    
    fig1.savefig('theoretical_C_naive.png', dpi=300, bbox_inches='tight')


    fig2, ax2 = plt.subplots(figsize=(8, 6))
    

    ax2.contourf(P, V, h_inner, levels=[0, np.inf], colors=['#B5E48C'], alpha=0.7)
 
    ax2.contour(P, V, h_inner, levels=[0], colors=['#386641'], linewidths=3)
    
    ax2.set_xlim(-20, 5)
    ax2.set_ylim(0, 15)
    ax2.set_xlabel('Position $p$ [m]')
    ax2.set_ylabel('Velocity $v$ [m/s]')
    
    inner_patch = mpatches.Patch(color='#B5E48C', label='Inner Safe Set $\mathcal{C}^*$ ($h^*(x) \geq 0$)')
    ax2.legend(handles=[inner_patch], loc='upper left')
    
    fig2.savefig('theoretical_C_inner.png', dpi=300, bbox_inches='tight')


    plt.show()

if __name__ == '__main__':
    generate_theoretical_plots()