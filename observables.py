import numpy as np
import itertools
import math
import scipy as sc
from scipy import integrate
import random as rand
from scipy import optimize

def total_momentum(system, beta, mu):

    P = 0

    for i in range(len(system.spval)):

        P += system.kfac * np.dot(system.kval[i],system.kval[i]) * fermi_factor(system.spval[i], mu, beta)

    return np.sqrt(P)

def constrained_f(x, system, beta):
    ''' Function for evaluating the total momentum and number
    of particles in the GC canonical ensemble. Pack these together
    to prevent the necesseity of two sum evaluations.

'''

    # Lagrange multiplier for the momentum.
    xi = x[:3]
    # Lagrange multiplier for the total number of particles (chemical potential).
    mu = x[3]

    k_i = system.kfac * system.kval

    P = [0] * 3
    N = 0

    for i in range(len(system.spval)):

        fermi_factor = 1.0 / (np.exp(-beta*(0.5*np.dot(k_i[i],k_i[i])-mu-np.dot(xi,k_i[i])))+1)
        P += k_i[i] * fermi_factor
        print P, k_i[i]
        N += fermi_factor

    print P, N
    P = P - system.total_K
    N = N - system.ne
    return ([P[0],P[1],P[2], N])

def centre_of_mass_obsv(system, beta):
    ''' Calculate centre of mass partition function and total energy.


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

    N = 0

    for speig in range(0,len(spval)):

        N = N + spval[speig][0]/(np.exp(-beta*(mu-spval[speig][1]))+1)

    return (2.0/pol)*N - ne

def nav_integral(eta, beta, integral_factor, ne):

    return integral_factor*np.power(beta,-1.5)*sc.integrate.quad(fermi_integrand, 0, np.inf, args=(0.5, eta))[0] - ne

def chem_pot_sum(system, evals, beta):

    return sc.optimize.fsolve(nav_sum, system.ef, args=(system.ne, evals, beta, system.pol))[0]

def chem_pot_integral(system, beta):

    return sc.optimize.fsolve(nav_integral, beta*system.ef, args=(beta, system.integral_factor, system.ne))[0] / beta

def fermi_integrand(x, nu, eta):

    return np.power(x,nu) / (np.exp(x-eta)+1)

def fermi_integrand_deriv(x, nu, eta):

    return np.power(x,nu) / (np.exp(x-eta)+2+np.exp(eta-x))

def fermi_integral(nu, eta):

    return sc.integrate.quad(fermi_integrand, 0, np.inf, args=(nu, eta))[0]

def energy_integral(beta, mu, integral_factor):

    return integral_factor * np.power(beta, -2.5) * fermi_integral(1.5, mu*beta)

def energy_sum(beta, mu, spval, pol):

    tenergy = 0

    for speig in range(0,len(spval)):
        tenergy = tenergy + (spval[speig][0]*spval[speig][1])/(np.exp(beta*(spval[speig][1]-mu))+1)

    return (2.0/pol)*tenergy

def hfx_integrand(eta):

    return fermi_integral(-0.5, eta)**2

def hfx_integral(system, beta, mu):

    hfx = sc.integrate.quad(hfx_integrand, -np.inf, beta*mu)[0]

    return (-system.L**3/(2.*system.pol*sc.pi**3*beta**2)) * hfx

def specific_heat(beta, mu, spval):

    spec_heat = []
    cv = 0

    for temp in range(0,len(beta)):
        for speig in spval:
            cv = cv + beta[temp]**2*(speig-mu[temp])*np.exp(-beta[temp]*(mu[temp]-speig))/(np.exp(-beta[temp]*(mu[temp] - speig))+1)**2

        spec_heat.append(cv)
        cv = 0

    return spec_heat

def partition_function(beta, mu, spval):

    Z = []

    p_func = 1

    for temp in range(0,len(beta)):
        for speig in spval:
            p_func *= (1 + np.exp(-beta[temp]*(speig-mu[temp])))

        Z.append(p_func)
        p_func = 1

    return Z

def fermi_factor(ek, mu, beta):

    return 1.0/(np.exp(beta*(ek-mu))+1)

def classical_ocp(system, Tmin, Tmax):
    ''' Evaluate the classical excess energy using the parametrised fit given by
    J. P. Hansen PRA 8, 6 1973.

'''

    a1 = -0.895929
    b1 = 4.666486
    a2 = 0.113406
    b2 = 13.67541
    a3 = -0.908728
    b3 = 1.890560
    a4 = -0.116147
    b4 = 1.027755

    theta = np.arange(Tmin, Tmax, 0.01)
    U = []

    for T in theta:

        gamma = 1.0/(system.rs*T)
        U_xc = 1.5 * gamma**1.5 * (a1/(b1+gamma)**0.5 + a2/(b2+gamma) + a3/(b3+gamma)**1.5 + a4/(b4+gamma)**2)
        U.append(U_xc)

    return (theta, U)

def hfx_sum(system, beta, mu):
    '''Evaluate the HF exchange contribution as a summation.

Patrams
-------
system : Class
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
                hfx += 1.0/np.dot(K-Q, K-Q)*fermi_factor(system.spval[k], mu, beta)*fermi_factor(system.spval[q], mu, beta)

    return -hfx / (system.L*sc.pi)

def canonical_partition_function(beta, spval, nel, kvecs, L):

    label = np.arange(0,len(spval))
    combs = list(itertools.combinations(label, nel))
    count = 0
    tenergy = np.zeros(len(beta))
    part = np.zeros(len(beta))

    for x in combs:
        index = list(x)
        if (list(sum(np.array(kvecs[index]))) == [0,0,0]):
            energy = spval[index].sum() + hf_potential(index, kvecs, L)
            count += 1
            for bval in range(0,len(beta)):
                exponent = np.exp(-beta[bval]*energy)
                tenergy[bval] += energy*exponent
                part[bval] += exponent

    print count, len(combs)
    return (tenergy, part, tenergy/part)

def madelung_constant(system):
    ''' Calculate the Madelung constant as described in Fraser er al. PRB 53 5 1814. Code copied from fortran version provided by James Shepherd.
    Here vm is given as
    ::math
        1/V sum_G exp(-\pi^2 G^2 / \kappa^2) / (pi*G^2) - \pi / (\kappa^2*V) + sum_R erfc(\kappa*|R|) / |R| - 2*\kappa / \sqrt(pi).
    In the code below we write
    a = - \pi / (\kappa^2*V)
    b = - 2*\kappa / \sqrt(pi).
    [todo] - explain parameters better / learn how to write maths in comments.
Params
------
system: class
    system parameters.

Returns
-------
vm: float
    Madelung constant.
---
'''

    omega = 1.0 / system.L**3
    # Taken from CASINO manual.
    kappa = 2.8 / omega**(1./3.)
    # a and b constants defined in function interface.
    a = -sc.pi / (kappa**2*system.L**3)
    b = -2.0*kappa / np.sqrt(sc.pi)

    kmax = 1
    rec_sum = 10
    rec_sum_new = 0

    while (abs(rec_sum-rec_sum_new) > system.de):
        rec_sum_new = rec_sum
        rec_sum = 0
        for i in range(-kmax,kmax):
            for j in range(-kmax,kmax):
                for k in range(-kmax,kmax):
                    dotprod = i**2 + j**2 + k**2
                    if (dotprod != 0):
                        Gsq = 1.0/system.L**2*dotprod
                        rec_sum += omega * (1.0/(sc.pi*Gsq)) * np.exp(-sc.pi**2*Gsq/kappa**2)
        kmax = kmax + 1
        if (kmax > 100):
            print "--------------------------------------------------"
            print "Not coverged with kmax = 100."
            print "Difference in last two iterations: ", np.abs(rec_sum-rec_sum_new)
            print "--------------------------------------------------"

    real_sum = 10
    real_sum_new = 0

    # [todo] - This is just repitition from above.
    while (abs(real_sum-real_sum_new) > system.de):
        real_sum_new = real_sum
        real_sum = 0
        for i in range(-kmax,kmax):
            for j in range(-kmax,kmax):
                for k in range(-kmax,kmax):
                    dotprod = i**2 + j**2 + k**2
                    if (dotprod != 0):
                        modr = system.L * dotprod
                        real_sum += math.erfc(kappa*modr) / modr
        kmax = kmax + 1
        if (kmax > 100):
            print "--------------------------------------------------"
            print "Not coverged with kmax = 100."
            print "Difference in last two iterations: ", np.abs(rec_sum-rec_sum_new)
            print "--------------------------------------------------"

    vm = real_sum + rec_sum + a + b
    #[todo] - Return the number of iterations?
    return vm

def madelung_approx(system):
    ''' Use expression in Schoof et al. (arxiv: 1502.04616) for the Madelung contribution to the total
    energy. This is a fit to the Fraser expression above.

Parameters
----------
system: class
    system being studied.

Returns
-------
E_M: float
    Madelung contriubtion to total energy (in Hartrees).

'''

    return - 0.5 * 2.837297 * (3.0/(4.0*sc.pi))**(1.0/3.0) * system.ne**(2.0/3.0) * system.rs**(-1.0)

def propagate_exact_spectrum(beta, eigv):

    Z = 0
    E_tot = 0

    for eig in eigv:

        exponent = np.exp(-beta*eig)
        Z += exponent
        E_tot += eig * exponent

    return E_tot / Z

def hf_potential(occ, kvecs, L):

    ex = 0.0

    for i in range(0,len(occ)):
        for j in range(i+1,len(occ)):
            q = kvecs[occ[i]] - kvecs[occ[j]]
            ex += 1.0/np.dot(q,q)

    return (-ex/(sc.pi*L))

def hfx_potential(spvals, kvecs, ki, beta, mu, L):

    ex = 0.0

    for kj in range(len(spvals)):

        if (kj != ki):
            q = kvecs[ki] - kvecs[kj]
            ex += fermi_factor(spvals[kj], mu, beta) / np.dot(q,q)

    return -ex/(sc.pi*L)

def check_self_consist(sp_new, sp_old):

    de = 0.0

    for i, j in zip(sp_new,sp_old): de += np.abs(i-j)

    return de

def hfx0_eigenvalues(system, beta, mu):

    sp_old = system.spval
    deg_new = np.array(system.deg_e)
    kinetic = system.spval
    kvecs = system.kval

    de = 1.
    att = 0
    mu_new = mu
    mu_it = 0
    while mu_it < 100:
        eig_it = 0
        mu_it += 1
        # Self-consistent loop for eigenvalues
        while  eig_it < 100:
            eig_it += 1
            sp_new = np.array([kinetic[ki]+hfx_potential(sp_old, kvecs, ki, beta, mu_new, system.L) for ki in range(len(kvecs))])
            de = check_self_consist(sp_new, sp_old)/len(kvecs)
            if (de < 1e-12):
                break
            sp_old = sp_new
        # Self-consistency condition for fermi-factors / chemical potential
        mu_old = chem_pot_sum(system, deg_new, beta)
        deg_new = system.compress_spval(sp_new)
        mu_new = chem_pot_sum(system, deg_new, beta)
        if (np.abs(mu_old-mu_new) < 1e-12):
            break

    ex = []
    mu = chem_pot_sum(system, deg_new, beta)

    return ((mu_it, eig_it), sp_new, mu)



def fthf_ex_energy(system, beta):

    e_0 = system.spval
    (nits, e_hf, mu) = hfx0_eigenvalues(system, beta, system.ef)

    kinetic = sum(e1*fermi_factor(e2, mu, beta) for (e1, e2) in zip(e_0, e_hf))
    potential = sum(hfx_potential(e_hf, system.kval, ki, beta, mu, system.L)*fermi_factor(e_hf[ki], mu, beta) for ki in range(0,len(e_0)))

    print kinetic, potential
    return kinetic + 0.5*potential

def sample_canonical_energy(system, beta, mu, nmeasure):

    p_i = np.array([fermi_factor(ek, mu, beta) for ek in system.spval])
    evals = system.spval
    E = np.zeros(nmeasure)

    en = 0
    Z = 0

    for it in range(0,nmeasure):

        (gen, orb_list) = create_orb_list(p_i, system.ne, system.M)
        if gen:
            E[Z] = sum(evals[orb_list])
            Z += 1

        if it % 1000 == 0 and Z > 0:
            print(it, np.mean(E[:Z]), np.std(E[:Z], ddof=1)/np.sqrt(Z))



def create_orb_list(probs, ne, M):

    selected_orbs = np.zeros(ne, dtype=np.int)
    nselect = 0

    for iorb in range(0,M):

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


def gc_part_func(sys, cpot, beta):

    Z_GC = 1.0

    for i in sys.spval:
        Z_GC = Z_GC * (1+np.exp(-beta*(i-cpot)))

    return Z_GC


def gc_potential(sys, cpot, beta):

    omega = 0.0

    for i in sys.spval:
        omega -= np.log(1+np.exp(-beta*(i-cpot)))

    return omega


def gc_potential_finite_basis(sys, cpot, beta):

    return 0

def free_energy(Z, beta):

    return -1.0/beta * np.log(Z)

def gc_correction_free_energy(sys, cpot, beta, delta, delta_error):

    # Assumption, Z_GC(N) / Z_GC = naccept/ntotal = delta, so -kT log(delta) = -kT log Z_GC(N) + kT log Z_GC
    # So -kT log Z_N = -kT log(delta) - kT log(Z_GC) + mu N, or F^0_N = F^0_GC + Delta(N) + mu N.

    muN = cpot * sys.ne
    F_GC = -(1.0/beta)*(np.log(gc_part_func(sys, cpot, beta)))
    Delta = -1.0/beta*np.log(delta)

    F_N = F_GC + Delta + muN
    F_N_error = delta_error / (beta*delta)

    return (F_N, F_N_error)


