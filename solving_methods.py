import numpy as np
import matplotlib.pyplot as plt
import qutip as qtp

from scipy.special import factorial
from joblib import Parallel, delayed
from itertools import product

def solve_JC_LME(initial_state, delta_c, delta_a, g_til, E, KAPPA, GAMMA, N_ph, t_in, t_fin, nt):
    '''
        Computes the evolution in time of the density matrix by using a LME with:
        - James-Cummings Hamiltonian in the drive frame
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

    return result



def solve_JC_LME_parallelized(initial_state, delta_c, delta_a, g_til, E_L, KAPPA_L, GAMMA_L, N_ph, t_in, t_fin, nt):
    '''
        Performs solve_JC_LME in a parallelized way over a (cartesian product of) list(s) of parameters.
    '''
    
    combinations = list(product(E_L, KAPPA_L, GAMMA_L))
    results = Parallel(n_jobs=4) (
        delayed(solve_JC_LME)(initial_state, delta_c, delta_a, g_til, E, KAPPA, GAMMA, N_ph, t_in, t_fin, nt) for E, KAPPA, GAMMA in combinations
    )
    return {params: resul for params, resul in zip(combinations, results)}



def solve_JC_LME_resonance(initial_state, omega_c, omega_a, g_til_L, E, KAPPA, GAMMA, N_ph, t_in, t_fin, nt):
    '''
        Performs solve_JC_LME upon varying the coupling constant g. 

        The difference with "solve_JC_LME_scan_omega" is that here both g and omega are recomputed
        to preserve resonance condition.
    '''
    omega_L = [omega_c - g_til for g_til in g_til_L]
    results = Parallel(n_jobs=4) (
        delayed(solve_JC_LME)(initial_state, omega_c-omega, omega_a-omega, g_til, E, KAPPA, GAMMA, N_ph, t_in, t_fin, nt) for (omega, g_til) in zip(omega_L, g_til_L)
    )
    return {params: results for params, results in zip(zip(omega_L, g_til_L), results)}




def solve_JC_LME_scan_omega(initial_state, omega_L, omega_c, omega_a, g_til, E, KAPPA, GAMMA, N_ph, t_in, t_fin, nt):
    '''
        Performs solve_JC_LME over a list of different values drive frequencies.
    '''
    results = Parallel(n_jobs=4) (
        delayed(solve_JC_LME)(initial_state, omega_c-omega, omega_a-omega, g_til, E, KAPPA, GAMMA, N_ph, t_in, t_fin, nt) for omega in omega_L
    )
    return {omega: result for omega, result in zip(omega_L, results)}
