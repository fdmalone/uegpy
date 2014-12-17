#!/usr/bin/env python
# -*- coding: utf-8 -*-

import ueg_sys as ueg_sys
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
    elif calc == 'test_root':
        data['Beta'] = beta
        (data['mu'], data['N']) = test_root(system, beta)
    elif calc == 'com':
        data['Beta'] = beta
        data['E_COM'] = [centre_of_mass_obsv(system, b)[0] for b in beta]

    return data

def main(args):
    ''' Main program

Params
------
args: list
    rs: float
        seitz radius.
    ne: integer
        number of electrons.
    pw: float
        plane wave cutoff measured in units of (2*pi/L)^2.
    pol: integer
        spin polarisation = 2 for fully polarised, 1 for unpolarised.
    calc_type: string
        possible calculation types include:
            All: perform all calculations
            partition: calculate the canonical partition function
            classical: calculate the classical excess energy of the one component plasma.
            test_root: something
            com: calculate the centre-of-mass correction to the total energy.
Returns
-------
data: pandas data frame
    contains columns of data for all requested quantities.
'''
    rs = float(args[0])
    ne = float(args[1])
    pw = float(args[2])
    pol = float(args[3])
    calc_type = args[4]

    system = ueg_sys.System(args)
    system.print_system_variables()
    data = run_calcs(system, calc_type)
    print data.to_string(index=False)

if __name__ == '__main__':

    main(sys.argv[1:])
