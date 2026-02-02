import numpy as np
import matplotlib.pyplot as plt
import qutip as qtp

def plot_occupations(results, selected_idx):
    '''
        Plots the occupations of different Fock states (i.e. the probabilities of having K photons in the 
        cavity for K = 0,...,N_ph) with different configurations of parameters.

    '''
    param_list = [
        r"Coupling $\tilde{g}$",
        r"Driving field $\mathcal{E}$",
        r"Cavity decay rate $\kappa$",
        r"Atom decay rate $\Gamma$"
    ]
    parameters = list(results.keys())
    densities  = list(results.values())

    fig, axes = plt.subplots(figsize=(12, 7), dpi=200)
    cbar = plt.get_cmap("viridis")

    axes.set_xlabel(r"Fock state $\vert n \rangle$")
    axes.set_ylabel("Occupation probability")

    for k, par in enumerate(parameters):
        rho_ss = densities[k].final_state.ptrace(1)
        occupations = rho_ss.diag()

        axes.plot(range(len(occupations)), occupations, color=cbar(k/len(parameters)), label=par[selected_idx], marker="o", ls="--", alpha=0.75)
        axes.set_xticks(np.arange(len(occupations)))

    axes.legend(title=f"{param_list[selected_idx]}")
    title = [param_list[i] + "=" + str(parameters[0][i]) for i in range(len(param_list)) if i != selected_idx]
    axes.set_title("\t ".join(title))
    axes.grid(color="lightgrey")

    #plt.tight_layout()
    plt.show()



def plot_two_photons_corr(result): 
    rho_ss = result.final_state.ptrace(1)
    N = rho_ss.shape[0]

    a = qtp.destruction()

