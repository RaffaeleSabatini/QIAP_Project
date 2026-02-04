import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import qutip as qtp
from scipy.special import factorial

plt.rcParams['axes.titlesize'] = 16     
plt.rcParams['axes.labelsize'] = 14     
plt.rcParams['xtick.labelsize'] = 12    
plt.rcParams['ytick.labelsize'] = 12    
plt.rcParams['legend.fontsize'] = 12    
plt.rcParams['figure.titlesize'] = 18   

def plot_mean_values(results, t_in, t_fin, nt, Gamma, title):
    times = np.linspace(t_in, t_fin, nt)
    fig, axes = plt.subplots(dpi=200, figsize=(9,7))
    cmap = plt.get_cmap("viridis")

    for k, (par, result) in enumerate(results.items()):
        color = cmap((k+0.5)/len(results))
        mean_N = result.expect[0]
        axes.plot(times, mean_N, label=par[0]/Gamma, color=color)
        axes.hlines(mean_N[-1], t_in, t_fin, linestyles="--", color=color)
    
    axes.grid("lightgrey")
    axes.set_xlabel('Time, t [sec]')
    axes.set_ylabel(r'Average photons number, $\langle \hat{a}^\dag \hat{a}\rangle$')
    axes.set_yticks(np.arange(10))
    axes.legend(title=rf"$\mathcal{{E}}/\Gamma$ = ", loc="center right", bbox_to_anchor=(0.98, 0.6))
    axes.set_title(title, weight="bold")
        
    plt.show()



def plot_occupations(results, selected_idx, param_list, with_poissonian_hist=False, prefix_title=""):
    '''
        Plots the occupations of different Fock states (i.e. the probabilities of having K photons in the 
        cavity for K = 0,...,N_ph) with different configurations of parameters.

    '''
    parameters = list(results.keys())
    densities  = list(results.values())

    fig, axes = plt.subplots(dpi=200, figsize=(9, 7))
    axes.grid(color="lightgrey") 
    cbar = plt.get_cmap("viridis")

    axes.set_xlabel(r"Fock state $\vert n \rangle$")
    axes.set_ylabel("Occupation probability")

    line_handles = []
    hist_handles = []

    for k, par in enumerate(parameters):
        # Plot occupations for different densities
        rho_ss = densities[k].final_state.ptrace(1)
        occupations = rho_ss.diag()
        N = len(occupations)
        N_dom = np.arange(N)

        color = cbar((k+0.5)/len(parameters))
        ln, = axes.plot(N_dom, occupations, color=color, label=f"{par[selected_idx]:.3f}", marker="o", ls="--")
        line_handles.append(ln)

        # Plot poissonian distributions to compare in the uncoupled case (g=0)
        if with_poissonian_hist:
            mean_N = densities[k].expect[0][-1]
            poisson_distr = np.exp(-mean_N) * mean_N**N_dom / factorial(N_dom)
            hst = plt.bar(N_dom, poisson_distr, label=fr"$\langle N \rangle$ = {mean_N:.3f}", color=color, alpha = 0.5)
            hist_handles.append(hst)

    
    axes.set_xticks(N_dom)

    leg1 = axes.legend(handles=line_handles ,title=f"{param_list[selected_idx]}", loc="upper right")
    axes.add_artist(leg1)
    if with_poissonian_hist and hist_handles:
        axes.legend(handles=hist_handles, title='Coherent (Poissonian) distribution with', loc="center right")

    title = [param_list[i] + "=" + str(parameters[0][i]) for i in range(len(param_list)) if i != selected_idx]
    axes.set_title(prefix_title + " - ".join(title), weight="bold")

    #plt.tight_layout()
    plt.show()



def plot_two_photons_corr(results, omega_res, g_til, GAMMA, KAPPA, E): 
    omega_L = np.array(list(results.keys()))
    N       = omega_L.shape[0]

    two_photons_corr = np.zeros(N)
    for k, res in enumerate(results.values()):
        rho_ss = res.final_state.ptrace(1)
        N = rho_ss.shape[0]
        a = qtp.destroy(N)
        two_photons_corr[k] = qtp.expect(a.dag()*a.dag()*a*a, rho_ss) / (qtp.expect(a.dag()*a, rho_ss))**2

    # Plot
    normalized_omega = (omega_L-omega_res)/g_til

    legend = [
        Line2D([0], [0], color="none", label=rf"$\tilde g=${g_til}"),
        Line2D([0], [0], color="none", label=rf"$\Gamma=${GAMMA}"),
        Line2D([0], [0], color="none", label=rf"$\kappa=${KAPPA}"),
        Line2D([0], [0], color="none", label=rf"$\mathcal{{E}}=${E}")
    ]

    fig, axes = plt.subplots(dpi=200, figsize=(9, 7))
    axes.plot(normalized_omega, two_photons_corr, "o-")
    axes.hlines(0, np.min(normalized_omega), np.max(normalized_omega), "red", "--")
    axes.set_xlabel(r"Relative drive frequency, $(\omega - \omega_{0})/\tilde{g}$")
    axes.set_xticks(normalized_omega)
    axes.set_ylabel(r"Two-photons correlation function, $g^{(2)}$")
    
    axes.grid("lightgrey")
    axes.set_title("Two-photons correlation at steady state around resonance", weight="bold")
    axes.legend(handles=legend, title="Parameters")

    plt.show()


