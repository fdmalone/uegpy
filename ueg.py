#!/usr/bin/env python

import sys
import scipy as sc
from scipy import integrate
import numpy as np
import matplotlib.pyplot as pl
import time
import itertools

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

    nmax = int((ecut/kfac)**(1/2.))

    spval = []
    vec = []
    kval = []

    for ni in range(-nmax, nmax+1):
        for nj in range(-nmax, nmax+1):
            for nk in range(-nmax, nmax+1):
                spe = 0.5*kfac**2*(ni**2 + nj**2 + nk**2)
                if (spe < ecut):
                    kval.append([ni,nj,nk])
                    spval.append(spe)

    return (spval, kval)

def compress_spval(spval):

    # Work out the degeneracy of each eigenvalue.
    j = 1
    i = 0
    it = 0
    deg = []

    while it < len(spval)-1:
        eval1 = spval[i]
        eval2 = spval[i+j]
        if eval2 == eval1:
            j += 1
        else:
            deg.append([j,eval1])
            i += j
            j = 1
        it += 1

    deg.append([j,eval1])

    return deg

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

def chem_pot_integral(integral_factor, efermi, de, ne):

    beta = 0.01
    db = beta
    T = 0

    mu = []
    xval = []
    npart = []
    iterations = int(5/db)

    for x in range(0,iterations):
        mu.append(newton_integral(integral_factor, beta*efermi, beta, de, ne)/beta)
        xval.append(beta)
        beta = beta + db

    return(xval, mu)

def newton_integral(integral_factor, mu, beta, de, ne):

    it = 0
    mu_new = 0
    npart = 0

    (nav, n_deriv) = nav_deriv_integral(integral_factor, beta, mu)
    mu_new = mu - (nav-ne)/n_deriv
    mu = mu_new

    while (abs(ne-npart) > de):
        (npart, n_deriv) = nav_deriv_integral(integral_factor, beta, mu)
        mu_new = mu - (npart-ne)/n_deriv
        mu = mu_new
        it = it + 1

    return mu

def newton(spval, ne, beta, efermi, de):

    mu = efermi
    it = 0
    mu_new = 0
    npart = 0

    (nav, n_deriv) = nav_deriv(spval, mu, beta)
    mu_new = mu - (nav-npart)/n_deriv
    mu = mu_new

    while (abs(npart-ne) > de):
        (npart, n_deriv) = nav_deriv(spval, mu_new, beta)
        mu_new = mu - (npart-ne)/n_deriv
        mu = mu_new
        it = it + 1

    return mu

def chem_pot(spval, ne, efermi, de):

    beta = 0.01
    db = beta
    T = 0

    mu = []
    xval = []
    npart = []
    iterations = int(5/db)

    for x in range(0,iterations):
        mu.append(newton(spval, ne, beta, efermi, de))
        xval.append(beta)
        beta = beta + db

    return(xval, mu)

def fermi_integrand(x, nu, eta):

    return np.power(x,nu) / (np.exp(x-eta)+1)

def fermi_integrand_deriv(x, nu, eta):

    return np.power(x,nu) / (np.exp(x-eta)+2+np.exp(eta-x))

def energy_integral(beta, mu, integral_factor):

    return integral_factor * np.power(beta, -2.5) * sc.integrate.quad(fermi_integrand, 0, np.inf, args=(1.5, mu*beta))[0]

def energy_integral_loop(beta, mu, integral_factor, pol):

    energy = []

    for x in range(0,len(beta)):
        energy.append(energy_integral(beta[x], mu[x], integral_factor))

    return energy

def energy(beta, mu, spval):

    tot_energy = []
    tenergy = 0

    for temp in range(0,len(beta)):
        for speig in range(0,len(spval)):
            tenergy = tenergy + (spval[speig][0]*spval[speig][1])/(np.exp(-beta[temp]*(mu[temp]-spval[speig][1]))+1)

        tot_energy.append(tenergy)
        tenergy = 0

    return tot_energy

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

def canonical_partition_function(beta, spval, nel, kval, K):

    label = np.arange(0,len(spval))
    combs = list(itertools.combinations(label, nel))
    count = 0
    tenergy = np.zeros(len(beta))
    part = np.zeros(len(beta))

    for x in combs:
        index = np.array(x)
        tk = sum(kval[index])
       # if np.array_equal(tk,K):
        count += 1
        print count, spval[index], spval[index].sum()
        energy = spval[index].sum()
        for bval in range(0,len(beta)):
            exponent = np.exp(-beta[bval]*energy)
            tenergy[bval] += energy*exponent
            part[bval] += exponent
        print tenergy[9], part[9]
    return (tenergy, part, count)

def print_file(beta, obsv, name):

    print "\t", "Beta", "\t", name
    for i in range(0,len(beta)-1):
        print "\t", beta[i], "\t", obsv[i]

    print "\n"

def main(args):

    rs = float(args[0])
    ne = float(args[1])
    ecut = float(args[2])
    pol = 2

    (L, kfac) = setup(rs, ne)
    # Fermi Momentum.
    k_f = kf(ne, L, pol)
    # Fermi energy.
    e_f = ef(ne, L, pol)
    integral_factor = L**3 * np.sqrt(2) / (pol*sc.pi**2)
    # Single particle energies.
    (spval, kval) = sp_energies(kfac, ecut)
    spval = np.array(spval)
    kval = [x for y, x in sorted(zip(spval, kval))]
    kval = np.array(kval)
    spval.sort()
    print spval
    deg = compress_spval(spval)
    (t_energy_fin, t_energy_inf) = total_energy_T0(spval, e_f, ne)
    beta = -10
    #for x in range(-10,10):
        #beta = beta + 1
        #print beta, nav_integral(integral_factor,  0.01, beta)
    ne = 1
    print "# of electrons: ", ne
    print "# rs: ", rs
    print "# System Length: ", L
    print "# Fermi wavevector for infinite system: ", k_f
    print "# Fermi energy for infinite system: ", e_f
    print "# Fermi energy for discrete system: ", dis_fermi(spval, ne)
    print "# kspace grid spacing: ", kfac
    print "# of plane waves: ", spval.size
    print "# Ground state energy for finite system: ", t_energy_fin
    print "# Ground state total energy for infinite system: ", t_energy_inf

    print nav(deg, ne, 0.1, -5.11688)
    start = time.time()
    (xval, mu) = chem_pot(deg, ne, dis_fermi(spval, ne), 1e-12)
    (xval, mu1) = chem_pot_integral(integral_factor, dis_fermi(spval, ne), 1e-6, ne)
    end = time.time()
    print "# Time taken to find chemical potential: ", end-start
    tenergy = [i for i in energy(xval, mu, deg)]
    tenergy2 = [i for i in energy_integral_loop(xval, mu1, integral_factor, pol)]
    beta = -4.0
    tenergy3 = np.zeros(len(xval))
    part = np.zeros(len(xval))
    counter = 0
    #for k in kval:
    result = canonical_partition_function(xval, spval, int(ne), kval, kval[0])
    tenergy3 += result[0]
    part += result[1]
    counter += result[2]

    NAV = []
    print counter
    for bval in range(0,len(xval)):
        NAV.append(nav(deg, ne, xval[bval], mu[bval]))
    #for x in range(-10,100):
        #beta = beta + 0.1
        #print beta, nav_integral(integral_factor,  0.1, 0.3)

    print_file(xval, tenergy, name="Energy")
    print_file(xval, tenergy2, name="Energy")
    print_file(xval, tenergy3/part, name="Energy")
    print_file(xval, mu, name="ChemPot")
    #print_file(xval, mu1, name="ChemPot")
    print_file(xval, NAV, name="N_av")

if __name__ == '__main__':

    main(sys.argv[1:])
