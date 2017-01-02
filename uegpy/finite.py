'''Evaluate properties of system with a finite particle number'''

import numpy as np
import scipy as sc
from scipy import optimize
from utils import fermi_factor
import itertools
import utils as ut
import dielectric as di


def centre_of_mass_obsv(system, beta):
    '''Calculate centre of mass partition function and total energy.

Parameters
----------
system : class
    System class containing system information.
beta : float
    Inverse temperature.

Returns
-------
E_tot : float
    Total energy contribution from centre of mass motion.
Z : float
    Centre of mass partition function.

'''

    Z = 0
    E_tot = 0

    for kval in system.kval:

        E_K = system.kfac**2/(2.0*system.ne)*np.dot(kval, kval)
        exponent = np.exp(-beta*E_K)
        Z += exponent
        E_tot += E_K * exponent

    return (E_tot/Z, Z)


def nav_sum(mu, ne, spval, beta, pol):
    '''Calculate average number of electrons.

Parameters
----------
mu : float
    chemical potential.
ne : int
    Number of electrons.
spval : list of lists
    Single particle eigenvalues and degeneracies.
beta : float
    Inverse temperature.
pol : int
    Polarisation.

Returns
-------
N : float
    Number of electrons.

'''

    N = sum(g_k*fermi_factor(e_k, mu, beta) for (g_k, e_k) in spval)

    return (2.0/pol) * N

def nav_deriv(mu, ne, spval, beta, pol):
    '''Calculate average number of electrons.

Parameters
----------
mu : float
    chemical potential.
ne : int
    Number of electrons.
spval : list of lists
    Single particle eigenvalues and degeneracies.
beta : float
    Inverse temperature.
pol : int
    Polarisation.

Returns
-------

N : float
    Number of electrons.
'''

    N = sum(beta*g_k/(2*(np.cosh(beta*(e_k-mu))+1))  for
            (g_k, e_k) in spval)

    return (2.0/pol) * N


def nav_diff(mu, ne, spval, beta, pol):
    '''Calculate difference between expected and average number of electrons.

Parameters
----------
mu : float
    chemical potential.
ne : int
    Number of electrons.
spval : lists of lists
    Single particle eigenvalues and degeneracies.
beta : float
    Inverse temperature.
pol : int
    Polarisation.

Returns
-------

Nav - ne : float
    Difference between expected and actual number of electrons.

'''

    return nav_sum(mu, ne, spval, beta, pol) - ne


def chem_pot_sum(system, eigs, beta):
    '''Find the chemical potential for finite system.

Parameters
----------
system : class
    System class containing system information.
beta : float
    Inverse temperature.

Returns
-------
mu : float
   Chemical potential.

'''
    return (
        sc.optimize.fsolve(nav_diff, system.ef, args=(system.ne, eigs,
                           beta, system.pol))[0]
    )


def energy_sum(beta, mu, spval, pol):
    '''Calculate internal energy for free electron gas.

Parameters
----------
beta : float
    Inverse temperature.
mu : float
    chemical potential.
spval : list of lists
    Single particle eigenvalues and degeneracies.
pol : int
    Polarisation.

Returns
-------

Nav - ne : float
    Difference between expected and actual number of electrons.

'''

    U = sum(g_k*e_k*fermi_factor(e_k, mu, beta) for (g_k, e_k) in spval)

    return (2.0/pol) * U


def hfx_sum(system, beta, mu):
    '''Evaluate the HF exchange contribution as a summation.

Parameters
----------
system : class
    system variables.
beta : float
    beta value
mu : float
    chemical potential at beta

Returns
-------
hfx : float
    hf exchange energy

'''

    hfx = 0

    # [todo] - Make this less stupid.
    for k in range(len(system.kval)):
        for q in range(k):
            K = system.kval[k]
            Q = system.kval[q]
            if not np.array_equal(K,Q):
                hfx += (
                   1.0/np.dot(K-Q, K-Q)*fermi_factor(system.spval[k], mu, beta)*
                                        fermi_factor(system.spval[q], mu, beta)
                )

    return -hfx / (system.L*sc.pi)


def hf_potential(occ, kvecs, L):
    '''Hartree-Fock exchange energy for given determinant

Parameters
----------
occ : list
    List of occupied single particle orbitals.
kvecs : list
    kvectors.
L : float
    Box lenght.

Returns
-------
ex : float
    Exchange energy.

'''

    ex = 0.0

    for i in range(0,len(occ)):
        for j in range(i+1,len(occ)):
            q = kvecs[occ[i]] - kvecs[occ[j]]
            ex += 1.0 / np.dot(q,q)

    return (-ex/(sc.pi*L))


def hfx_potential(spvals, kvecs, ki, beta, mu, L):
    '''Finite temperature Hartree-Fock potential.

Parameters
----------
system : class
    System begin studied.
ki : list
    kvector associated with potential.
beta : float
    Inverse temperature.
mu : float
    Chemical potential.

Returns
-------
ex : float
    exchange potential.

'''

    ex = 0.0

    for kj in range(len(spvals)):

        if (kj != ki):
            q = kvecs[ki] - kvecs[kj]
            ex += fermi_factor(spvals[kj], mu, beta) / np.dot(q,q)

    return -ex/(sc.pi*L)


def canonical_partition_function(system, beta):
    '''Calculate canonical partition function from single particle eigenvalues.

    Warning: This gets very expensive very fast.

Parameters
----------
system : class
    system class.

Returns
-------
U : float
    Internal energy.
Z : float
    Canonical partition function.

'''

    label = np.arange(0,len(system.spval))
    combs = list(itertools.combinations(label, system.ne))
    U = 0
    Z = 0

    for x in combs:
        index = list(x)
        E_i = system.spval[index].sum()
        exponent = np.exp(-beta*E_i)
        Z += exponent
        U += E_i * exponent

    return (U/Z, Z)



def fthf_self_consistency(system, beta, mu):
    ''' Run self consistency loop to find finite temperature Hartree-Fock
    eigenvalues and chemical potential.

    Todo: Check this makes sense + document better, integrate with Monte Carlo.

Parameters
----------
system : class
    System being studied.
beta : float
    Inverse Temperature.
mu : float
    Chemical potential.

Returns
-------
iterations : tuple
    Number of iterations required to find self consistency of mu and single
    particle eigenvalues.
sp_x : list
    Single particle eigenvalues.
mu_x : float
    Hartree-Fock chemical potential.

'''

    sp_old = system.spval
    deg_new = np.array(system.deg_e)
    kinetic = system.spval
    kvecs = system.kval

    de = 1.
    att = 0
    mu_x = mu
    mu_it = 0
    while mu_it < 100:
        eig_it = 0
        mu_it += 1
        # Self-consistent loop for eigenvalues
        while  eig_it < 100:
            eig_it += 1
            sp_new = np.array([kinetic[ki]+hfx_potential(sp_old, kvecs, ki,
                        beta, mu_x, system.L) for ki in range(len(kvecs))])
            de = check_self_consist(sp_new, sp_old)/len(kvecs)
            if (de < 1e-6):
                break
            sp_old = sp_new
        # Self-consistency condition for fermi-factors / chemical potential
        mu_old = chem_pot_sum(system, deg_new, beta)
        deg_new = system.compress_spval(sp_new)
        mu_x = chem_pot_sum(system, deg_new, beta)
        if (np.abs(mu_old-mu_x) < 1e-6):
            break

    ex = []
    mu = chem_pot_sum(system, deg_new, beta)

    return ((mu_it, eig_it), sp_new, mu)



def fthf_ex_energy(system, beta):
    ''' Evaluate finite temperature Hartree-Fock internal energy.

    Todo : notes + grand canonical equivalent.

Parameters
----------
system : class
    System being studied.
beta : float
    Inverse Temperature.

Returns
-------
U_tx : float
    Finite temperature Hartree-Fock internal energy.

'''

    e_0 = system.spval
    (nits, e_hf, mu) = fthf_self_consistency(system, beta, system.ef)

    kinetic = sum(e1*fermi_factor(e2, mu, beta) for (e1, e2) in zip(e_0, e_hf))
    potential = sum(hfx_potential(e_hf, system.kval, ki, beta, mu,
                    system.L)*fermi_factor(e_hf[ki], mu, beta)
                    for ki in range(0,len(e_0)))

    return kinetic + 0.5*potential


def check_self_consist(sp_new, sp_old):
    '''Check self consistency for array

Parameters
----------
sp_new : list
    new single particle eigenvalues.
sp_old : list
    old single particle eigenvalues.

Returns
-------
de : float
    cumulative difference between new and old eigenvalues.

'''

    de = 0.0

    for i, j in zip(sp_new,sp_old): de += np.abs(i-j)

    return de


def gc_part_func(sys, cpot, beta):
    ''' Grand canonical partition function for finite system.

    .. math::
        Z_{GC} = \prod_i (1 + e^{-\\beta(e_i-\\mu)})


Parameters
----------
system : class
    System being studied.
beta : float
    Inverse Temperature.

Returns
-------
Z_GC : float
    Grand canonical partition function.

'''

    Z_GC = 1.0

    for i in sys.spval:
        Z_GC = Z_GC * (1+np.exp(-beta*(i-cpot)))

    return Z_GC


def rpa_xc_free_energy(sys, mu, beta, lmax):

    def integrand(q, sys, mu, beta, lmax):

        #print q, ekq, mu, beta, di.lindhard_matsubara_finite(sys, ekq, mu, beta, 1)
        #print "TYPE: ", di.lindhard_matsubara_finite(sys, ekq, mu, beta, 1)
        return (
            1.0/(sys.rho*beta) *
            sum([np.log(1.0-ut.vq_vec(q)*
                di.lindhard_matsubara_finite(sys, q, mu, beta, l))
                for l in range(-lmax, lmax)]) + ut.vq_vec(q)
        )


    f_xc = sum([integrand(q, sys, mu, beta, lmax) for q in
                sys.kfac*sys.kval[1:]])

    return - 0.5 / (2*sc.pi)**3.0 * f_xc


def rpa_correlation_free_energy(sys, mu, beta, lmax):

    def integrand(q, sys, mu, beta, l):

        I = 0.0
        for l in range(-lmax, lmax+1):

            eps_0 = ut.vq_vec(sys.kfac*q) * di.lindhard_matsubara_finite(sys, q, mu, beta, l)
            print l, q, eps_0, I
            I += np.log(1-eps_0) + eps_0

        return I


    f_c = sum(integrand(q, sys, mu, beta, lmax) for q in sys.kval[1:])

    print beta, f_c
    return (0.5/(sys.ne*beta)) * f_c


def hfx_matsubara(sys, mu, beta, lmax):

    v_x = sum(ut.vq_vec(sys.kfac*q)*(sq_finite(q, sys, mu, beta, lmax)-1) for q
              in sys.kval[1:])

    return sys.ne / (2.0*sys.L**3.0) * v_x
