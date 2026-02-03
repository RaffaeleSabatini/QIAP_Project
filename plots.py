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



def plot_two_photons_corr(results, omega_res, g_til): 
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

    fig, axes = plt.subplots(dpi=200, figsize=(12, 7))
    axes.plot(normalized_omega, two_photons_corr, "o-")
    axes.hlines(0, np.min(normalized_omega), np.max(normalized_omega), "red", "--")
    axes.set_xlabel(r"Relative drive frequency, $(\omega - \omega_{0})/\tilde{g}$")
    axes.set_ylabel(r"Two-photons correlation function, $g^{(2)}$")
    axes.grid("lightgrey")

    plt.show()


