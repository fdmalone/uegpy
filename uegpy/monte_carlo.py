'''Simple Monte Carlo routines for evaluating in the canonical ensemble.'''
import utils as ut
import numpy as np
import random as rand
import finite as fn
import pandas as pd
import time


def sample_canonical_energy(system, beta, nmeasure):
    ''' Sample canonical energy of a non-interacting system.

    Properties of the non-interacting canonical system can be evaluated by
    simply discarding configurations generated in grand canonical ensemble whose
    particle number is not equal to the expected number of particles.

Parameters
----------
system : class
    System being studied.
beta : float
    Inverse temperature.
nmeasure : int
    Number of attempted Monte Carlo moves.
Returns
-------
frame : :class:`pandas.DataFrame`
    Frame containing estimates for properties of system at temperature theta.
'''

    mu = fn.chem_pot_sum(system, system.deg_e, beta)
    p_i = np.array([ut.fermi_factor(ek, mu, beta) for ek in system.spval])
    evals = system.spval
    ncycles = nmeasure / 10
    T = np.zeros(ncycles)
    V = np.zeros(ncycles)

    cycle = 0
    T_loc = 0
    V_loc = 0
    it = 0

    # Monte Carlo can take a while.
    start = time.time()
    while (it < nmeasure):

        (gen, orb_list) = create_orb_list(p_i, system.ne)
        if gen:
            T_loc += sum(evals[orb_list])
            V_loc += fn.hf_potential(orb_list, system.kval, system.L)
            it += 1
            if it % 10 == 0:
                T[cycle] = T_loc / 10
                V[cycle] = V_loc / 10
                T_loc = 0
                V_loc = 0
                cycle += 1


    end = time.time()

    frame = pd.DataFrame({'t': T, 'v': V})
    frame['u'] = frame['t'] + frame['v']
    means = frame.mean().to_frame().transpose()
    means['ne'] = system.ne
    std_err = np.sqrt(frame.var()/ncycles).to_frame().transpose()
    std_err['ne'] = system.ne
    new = pd.merge(means, std_err, on=['ne'], suffixes=('', '_error'))
    new = ut.add_mad(system, new)
    new['rs'] = system.rs
    new['M'] = len(system.spval)
    new['Beta'] = beta * system.ef

    return (new[sorted(new.columns.values)], end-start)


def create_orb_list(probs, ne):
    '''Create orbital list with :math:`N` electrons

Parameters
----------
probs : list
    Single particle probabilities (fermi factors).
ne : int
    Number of electrons.

Returns
-------
gen : boolean
    True if configuration with ne electrons was generated.
selected_orbs : list
    Selected orbitals.
'''

    selected_orbs = np.zeros(ne, dtype=np.int)
    nselect = 0

    for iorb in range(0,len(probs)):

        r = rand.random()

        if (probs[iorb] > r):
            nselect += 1
            if (nselect > ne):
                gen = False
                break
            selected_orbs[nselect-1] = iorb

    if (nselect == ne):
        gen = True
    else:
        gen = False

    return (gen, selected_orbs)

def gc_correction_free_energy(sys, cpot, beta, delta, delta_error):
    '''Canonical correction to free electron grand canonical partition function.
    Assumption, :math:`Z_{GC}(N) / Z_{GC} = \\delta`, so

    .. math::
        -kT \\log\\delta = -kT \\log Z_{GC}(N) + kT \\log Z_{GC}.

    Therefore,

    .. math::
        -kT \\log Z_N = -kT \\log \\delta - kT \\log Z_{GC} + \\mu N,

    or

    .. math::
        F^0_N = \\Omega + \\Delta(N) + \\mu N.

Parameters
----------
sys : class
    System being studied.
mu : float
    Chemical potential.
beta : float
    Inverse temperature.
delta : float
    naccept/ntotal.
delta_error : float
    standard error in delta.
Returns
-------
F_N : float
    Canonical free electron Helmholtz free energy.
'''

    muN = cpot * sys.ne
    F_GC = -(1.0/beta)*(np.log(gc_part_func(sys, cpot, beta)))
    Delta = -1.0/beta*np.log(delta)

    DF_N = F_GC + muN
    F_N = DF_N + Delta
    F_N_error = delta_error / (beta*delta)

    return (F_N, F_N_error)
