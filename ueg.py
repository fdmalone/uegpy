#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import scipy as sc
from scipy import integrate
from scipy import optimize
import numpy as np
import matplotlib.pyplot as pl
import time
import pandas as pd
import itertools
import math

def kf(ne, L, ns):

    k_f = (3*ns*sc.pi**2*ne/L**3)**(1/3.0)

    return k_f

def ef(ne, L, ns):

    e_f = 0.5*(3*ns*sc.pi**2*ne/L**3)**(2/3.)

    return e_f

def setup(rs, ne):

    # System Lenght.
    L = rs*(4*ne*sc.pi/3.)**(1/3.)
    # k = 2*pi/L n
    kfac = 2*sc.pi/L

    return (L, kfac)

def total_energy_T0(spval, E_f, ne):

    return (sum(spval[:ne]), 3*ne/5.*E_f)

def sp_energies(kfac, ecut):

    # Scaled Units to match with HANDE.
    # So ecut is measured in units of 1/kfac^2.
    nmax = int(math.ceil(np.sqrt((2*ecut))))

    spval = []
    vec = []
    kval = []

    for ni in range(-nmax, nmax+1):
        for nj in range(-nmax, nmax+1):
            for nk in range(-nmax, nmax+1):
                spe = 0.5*(ni**2 + nj**2 + nk**2)
                if (spe <= ecut):
                    kval.append([ni,nj,nk])
                    # Reintroduce 2 \pi / L factor.
                    spval.append(kfac**2*spe)

    # Sort the arrays in terms of increasing energy.
    spval = np.array(spval)
    kval = [x for y, x in sorted(zip(spval, kval))]
    kval = np.array(kval)
    spval.sort()

    return (spval, kval)

def compress_spval(spval, kval):

    # Work out the degeneracy of each eigenvalue.
    j = 1
    i = 0
    it = 0
    deg_e = []
    deg_k = []

    while it < len(spval)-1:
        eval1 = spval[i]
        eval2 = spval[i+j]
        if eval2 == eval1:
            j += 1
        else:
            deg_e.append([j,eval1])
            deg_k.append([j,kval[i]])
            i += j
            j = 1
        it += 1

    deg_e.append([j,eval1])
    deg_k.append([j,kval[i]])

    return (deg_e, deg_k)

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

def constrained_canonical_ensemble(system, beta):
    ''' Attempt to find the chemical potential and momentum lagrange
    multiplier which will result in a Grand Canonical partition function (GCPF)
    restricted to the N = ne, P = total_K subspace. Typically we are interested
    in evaluating everything in the total_K = 0 subspace for comparison with DMQMC
    calculations.

    This doesn't work as the total momentum is automatically zero in the GC ensemble
    by symmetry.....

    The GC density matrix can be written

        \\rho = 1 / Z exp(-beta(\\hat{H} - \\mu \\hat{N} - \\vec{\\xi}\cdot\hat{K}),

    where \mu and \\vec{\\xi} are Lagrange multipliers which fix the average total
    number of particles and momentum respectively.

    For a non-interacting system the partition function, Z, can be shown to read

        Z = \\prod_i (1 + exp(\\beta(\\varepsilon_i - mu - \\vec{\\xi}\\cdot\\vec{p}_i)),

    and now \\varepsilon_i and \\vec{p}_i are single particle energies and the associated
    momentum.

    To work in a constrained (both by particle number and total momentum) space one needs to find
    the values of \\mu and \\vec{\\xi} such that

        < \\hat{N} > = \\sum_i 1 / (exp(beta(\\varepsilon - \\mu - \\vec{\\xi}\\cdot\\vec{p}_i))+1) = N_e

    and
        < \\hat{K} >  = \\sum_i 1 / (exp(beta(\\varepsilon - \\mu  - \\vec{\\xi}\\cdot\\vec{p}_i))+1) = K_{desired}.

    This amounts to finding the roots of the function

        F = (< \\hat{K} > - K_desired, < \\hat{N} > - N_e),

    which can be achieved using "standard" root finding procedures. Here scipy.optimize.fsolve is used
    which apparently employs MINPACK’s hybrd and hybrj algorithms.
'''

    # Initial guess for the momentum lagrange multiplier
    # and chemical potential.
    x0 = [0, 0, 0, system.ef]
    xi = []
    mu = []
    N = []

    for b in beta:
        # Find the roots of the function constrained_f.
        solution = sc.optimize.root(constrained_f, x0, args=(system, b))
        x0[0:3] = solution[0:3]
        xi.append(solution[0:3])
        mu.append(solution[3])
        N.append(nav_constrained(system, b, solution[3], solution[0:3]))


    return (xi, mu, N)

def test_root(system, beta):

    x0 = system.ef
    mu = []
    N = []

    for b in beta:
        solution = sc.optimize.fsolve(n_test, x0, args=(system,beta))[0]
        print solution
        #mu.append(solution)
        #N.append(n_test(solution, system, b))

    return (mu, N)

def n_test(mu, system, beta):

    N = 0.

    for i in range(len(system.spval)):

        N += 1.0 / (np.exp(beta*(system.spval[i]-mu))+1.0)

    return N - system.ne

def nav_constrained(system, beta, mu, xi):

    N = 0

    for i in range(len(system.spval)):

        N += 1.0 / (np.exp(beta*(system.spval[i]-mu-np.dot(xi,system.kval[i])))+1)

    return N

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


def dis_fermi(spval, ne):

    return 0.5*(spval[ne-1]+spval[ne])

def nav_deriv(spval, x, beta):

    deriv = 0
    N = 0

    for speig in range(0,len(spval)):

        fermi_factor =  spval[speig][0]/(np.exp(beta*(spval[speig][1]-x))+1)
        N = N + fermi_factor
        deriv = deriv + beta*np.exp(beta*(spval[speig][1]-x))*fermi_factor**2

    return (N, deriv)

def nav(spval, ne, beta, guess):

    mu = guess
    N = 0
    it = 0

    for speig in range(0,len(spval)):

        N = N + spval[speig][0]/(np.exp(-beta*(mu-spval[speig][1]))+1)

    return N

def nav_integral(integral_factor, beta, eta):

    #print sc.integrate.quad(fermi_integrand,np.inf, args=(0.5,eta)), integral_factor, np.power(beta,-1.5)
    return integral_factor * np.power(beta,-1.5)  * sc.integrate.quad(fermi_integrand, 0, np.inf, args=(0.5, eta))[0]

def nav_deriv_integral(integral_factor, beta, eta):

    N = sc.integrate.quad(fermi_integrand, 0, np.inf, args=(0.5, eta))[0]
    Nderiv = sc.integrate.quad(fermi_integrand_deriv, 0, np.inf, args=(0.5, eta))[0]

    return (integral_factor*np.power(beta,-1.5)*N, integral_factor*np.power(beta,-1.5)*Nderiv)

def chem_pot_newton_integral(system, beta):

    mu = beta*system.ef
    it = 0
    mu_new = 0
    npart = 0
    ne = system.ne

    (nav, n_deriv) = nav_deriv_integral(system.integral_factor, beta, mu)
    mu_new = mu - (nav-ne)/n_deriv
    mu = mu_new

    while (abs(ne-npart) > system.root_de):
        (npart, n_deriv) = nav_deriv_integral(system.integral_factor, beta, mu)
        mu_new = mu - (npart-ne)/n_deriv
        mu = mu_new
        it = it + 1

    return mu / beta

def chem_pot_newton_sum(system, beta):

    mu = system.ef
    it = 0
    mu_new = 0
    npart = 0
    ne = system.ne

    spval = system.deg_e
    (nav, n_deriv) = nav_deriv(spval, mu, beta)
    mu_new = mu - (nav-npart)/n_deriv
    mu = mu_new

    while (abs(npart-ne) > system.root_de):
        (npart, n_deriv) = nav_deriv(spval, mu_new, beta)
        mu_new = mu - (npart-ne)/n_deriv
        mu = mu_new
        it = it + 1

    return mu

def fermi_integrand(x, nu, eta):

    return np.power(x,nu) / (np.exp(x-eta)+1)

def fermi_integrand_deriv(x, nu, eta):

    return np.power(x,nu) / (np.exp(x-eta)+2+np.exp(eta-x))

def energy_integral(beta, mu, integral_factor):

    return integral_factor * np.power(beta, -2.5) * sc.integrate.quad(fermi_integrand, 0, np.inf, args=(1.5, mu*beta))[0]

def energy(beta, mu, spval):

    tenergy = 0

    for speig in range(0,len(spval)):
        tenergy = tenergy + (spval[speig][0]*spval[speig][1])/(np.exp(beta*(spval[speig][1]-mu))+1)

    return tenergy

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

def hartree0_sum(system, beta, mu):
    '''Evaluate the HF0 energy contribution as a summation.

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
hfx0 : float
    hf0 exchange energy

'''

    hfx0 = 0

    # [todo] - Make this less stupid.
    for k in range(len(system.kval)):
        for q in range(k):
            K = system.kval[k]
            Q = system.kval[q]
            if not np.array_equal(K,Q):
                hfx0 += 1.0/np.dot(K-Q, K-Q)*fermi_factor(system.spval[k], mu, beta)*fermi_factor(system.spval[q], mu, beta)

    return -hfx0 / (system.L*sc.pi)

def canonical_partition_function(beta, spval, nel, kval, K):

    label = np.arange(0,len(spval))
    combs = list(itertools.combinations(label, nel))
    count = 0
    tenergy = np.zeros(len(beta))
    part = np.zeros(len(beta))

    for x in combs:
        index = np.array(x)
        tk = sum(kval[index])
        if np.array_equal(tk,K):
            count += 1
            energy = spval[index].sum()
            for bval in range(0,len(beta)):
                exponent = np.exp(-beta[bval]*energy)
                tenergy[bval] += energy*exponent
                part[bval] += exponent

    print count
    return (tenergy, part, count)

def print_file(beta, obsv, name):

    print "\t", "Beta", "\t", name
    for i in range(0,len(beta)-1):
        print "\t", beta[i], "\t", obsv[i]

    print "\n"

class System:

    def __init__(self, args):

        # Seitz radius.
        self.rs = float(args[0])
        # Number of electrons.
        self.ne = int(args[1])
        # Kinetic energy cut-off.
        self.ecut = float(args[2])
        # Spin polarisation.
        self.pol = int(args[3])
        # Box Length.
        self.L = self.rs*(4*self.ne*sc.pi/3.)**(1/3.)
        # k-space grid spacing.
        self.kfac = 2*sc.pi/self.L
        # Fermi Wavevector (infinite system).
        self.kf = (3*self.pol*sc.pi**2*self.ne/self.L**3)**(1/3.)
        # Fermi energy (inifinite systems).
        self.ef = 0.5*self.kf**2
        # Integral Factor.
        self.integral_factor = self.L**3 * np.sqrt(2) / (self.pol*sc.pi**2)
        # Single particle eigenvalues and corresponding kvectors
        (self.spval, self.kval) = sp_energies(self.kfac, self.ecut)
        # Compress single particle eigenvalues by degeneracy.
        (self.deg_e, self.deg_k) = compress_spval(self.spval, self.kval)
        # Number of plane waves.
        self.M = len(self.spval)
        (self.t_energy_fin, self.t_energy_inf) = total_energy_T0(self.spval, self.ef, self.ne)
        self.ef_fin = dis_fermi(self.spval, self.ne)
        # Change this to be more general, selected to be the gamma point.
        self.total_K = self.kval[0]
        self.beta_max = 5.0
        self.root_de = 1e-10

    def print_system_variables(self):

        print " # Number of electrons: ", self.ne
        print " # Spin polarisation: ", self.pol
        print " # rs: ", self.rs
        print " # System Length: ", self.L
        print " # Fermi wavevector for infinite system: ", self.kf
        print " # kspace grid spacing: ", self.kfac
        print " # Number of plane waves: ", self.spval.size
        print " # Fermi energy for infinite system: ", self.ef
        print " # Fermi energy for discrete system: ", self.ef_fin
        print " # Ground state energy for finite system: ", self.t_energy_fin
        print " # Ground state total energy for infinite system: ", self.t_energy_inf

def run_calcs(system, calc='All'):
    '''Run user defined calculations.

Parameters
----------
system: class object
    System parameters
calc: list of strings
    What calculations to run on system

Returns
-------
data : pandas data frame containing desired quantities.
'''

    data = pd.DataFrame()
    beta = np.arange(0.1/system.ef, system.beta_max, 0.1/system.ef)
    data['Beta'] = beta
    data['T/T_F'] = 1.0 / (system.ef*beta)
    data['M'] = system.M
    calc_time = []

    if calc == 'All':
        start = time.time()
        # Find the chemical potential.
        mu = [chem_pot_newton_sum(system, b) for b in beta]
        data['chem_pot'] = mu
        # Evaluate observables.
        beta_mu = zip(beta, mu)
        #for b, m in beta_mu:
            #print b, m, energy(b, m, system.deg_e), nav(system.deg_e, system.ne, b, m)
        data['Energy_sum'] = [energy(b, m, system.deg_e) for b, m in beta_mu]
        #data['HF0_sum'] = [hartree0_sum(system, b, m) for b, m in beta_mu]
        end = time.time()
        calc_time.append(end-start)
        start = time.time()
        mu = [chem_pot_newton_integral(system, b) for b in beta]
        beta_mu = zip(beta, mu)
        data['Energy_integral'] = [energy_integral(b, m, system.integral_factor) for b, m in beta_mu]
        data['E_COM'] = [centre_of_mass_obsv(system, b)[0] for b in beta]
        data['Diff'] = data['Energy_sum'] - data['E_COM']
        #(tenergy, partition, count) = canonical_partition_function(beta, system.spval, system.ne, system.kval, system.kval[0])
        #data['Partition'] = tenergy / partition
        end = time.time()
        calc_time.append(end-start)
        print " # Time taken for calculation: ", calc_time
    elif calc == 'partition':
        xval = np.arange(0,5,0.1)
        (tenergy, partition, count) = canonical_partition_function(xval, system.spval, system.ne, system.kval, system.kval[0])
        data['Beta'] = xval
        data['Partition'] = tenergy / partition
    elif calc == 'classical':
        (T, Uxc) = classical_ocp(system, 0.01, 10)
        data['T'] = T
        data['Theta'] = T / system.ef
        data['Beta'] = 1 / T
        data['Classical_Uxc'] = Uxc
    elif calc == 'constrained':
        data['Beta'] = beta
        (data['xi'], data['mu'], data['N']) = constrained_canonical_ensemble(system, beta)
    elif calc == 'test_root':
        data['Beta'] = beta
        (data['mu'], data['N']) = test_root(system, beta)
    elif calc == 'com':
        data['Beta'] = beta
        data['E_COM'] = [centre_of_mass_obsv(system, b)[0] for b in beta]

    return data

def main(args):

    rs = float(args[0])
    ne = float(args[1])
    pw = float(args[2])
    pol = float(args[3])
    calc_type = args[4]

    system = System(args)
    system.print_system_variables()
    data = run_calcs(system, calc_type)
    print data.to_string(index=False)

if __name__ == '__main__':

    main(sys.argv[1:])
