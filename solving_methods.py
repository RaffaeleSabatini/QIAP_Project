import numpy as np
import matplotlib.pyplot as plt
import qutip as qtp

from scipy.special import factorial
from joblib import Parallel, delayed
from itertools import product

def solve_JC_LME(initial_state, delta_c, delta_a, g_til, E, KAPPA, GAMMA, N_ph, t_in, t_fin, nt, show_N_avg=True):
    '''
        Computes the evolution in time of the density matrix by using a LME with:
        - James-Cumming Hamiltonian in the drive frame
        - destruction operator "a" and sigma-minus as Lindblad operators 
        - KAPPA and GAMMA as respectively the (squared) decay rates.

        ------------------------------
        ARGUMENTS:
            - delta_c: cavity detuning (w_c - w)
            - delta_a: atom detuning (w_a - w)
            - g_til  : atom-cavity coupling
            - E      : pumping field amplitude
            - KAPPA  : photon decay rate
            - GAMMA  : atom decay rate
            - N_ph   : light Hilbert space dimension

        If show_N_avg is true, it also plots the time evolution of <N>.
    '''
        
    id2 = qtp.qeye(2)
    idN = qtp.qeye(N_ph)

    # Building James-Cummings hamiltonian
    exc_state = qtp.fock(2, 0)
    a = qtp.tensor(id2, qtp.destroy(N_ph))
    sigma_p = qtp.tensor(qtp.sigmap(), idN)
    e_proj = qtp.tensor(exc_state*exc_state.dag(), idN)

    H = delta_c*a.dag()*a + delta_a*e_proj + g_til*(sigma_p*a + sigma_p.dag()*a.dag()) + E*(a+a.dag())

    # Decay operators of the bath
    a       = qtp.tensor(id2, qtp.destroy(N_ph))
    sigma_m = qtp.tensor(qtp.sigmam(), idN)
    c_ops = [
        np.sqrt(KAPPA)*a,
        np.sqrt(GAMMA)*sigma_m
    ]
    
    times = np.linspace(t_in, t_fin, nt)
    result = qtp.mesolve(H, initial_state, times, c_ops, e_ops=[a.dag()*a], options={'store_final_state':True})

    # Plotting results
    if show_N_avg:
        plt.figure(figsize=(12, 7),dpi=200)
        plt.plot(times, result.expect[0], label=r"$\langle \hat{a}^\dag \hat{a}\rangle$")
        plt.xlabel('Time t [sec]')
        plt.ylabel('Expectation value')
        plt.legend()
        plt.show()
    
    return result


def solve_JC_LME_parallelized(initial_state, delta_c, delta_a, g_til_L, E_L, KAPPA_L, GAMMA_L, N_ph, t_in, t_fin, nt):
    '''
        Performs solve_JC_LME in a parallelized way over a (cartesian product of) list(s) of parameters.
    '''
    
    combinations = list(product(g_til_L, E_L, KAPPA_L, GAMMA_L))
    results = Parallel(n_jobs=4) (
        delayed(solve_JC_LME)(initial_state, delta_c, delta_a, g_til, E, KAPPA, GAMMA, N_ph, t_in, t_fin, nt, False) for g_til, E, KAPPA, GAMMA in combinations
    )
    return {params: resul for params, resul in zip(combinations, results)}