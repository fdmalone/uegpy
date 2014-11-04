#!/usr/bin/env python

import sys
import scipy as sc
from scipy import integrate
import numpy as np
import matplotlib.pyplot as pl
import time

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
            tenergy = tenergy + spval[speig][0]*spval[speig][1]/(np.exp(-beta[temp]*(mu[temp]-spval[speig][1]))+1)

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
    spval.sort()
    deg = compress_spval(spval)
    (t_energy_fin, t_energy_inf) = total_energy_T0(spval, e_f, ne)
    beta = -10
    #for x in range(-10,10):
        #beta = beta + 1
        #print beta, nav_integral(integral_factor,  0.01, beta)

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

    start = time.time()
    (xval, mu) = chem_pot(deg, ne, dis_fermi(spval, ne), 1e-6)
    (xval, mu1) = chem_pot_integral(integral_factor, dis_fermi(spval, ne), 1e-6, ne)
    end = time.time()
    print "# Time taken to find chemical potential: ", end-start
    tenergy = [i for i in energy(xval, mu, deg)]
    tenergy2 = [i for i in energy_integral_loop(xval, mu1, integral_factor, pol)]
    beta = -4.0
    #for x in range(-10,100):
        #beta = beta + 0.1
        #print beta, nav_integral(integral_factor,  0.1, 0.3)

    print_file(xval, tenergy, name="Energy")
    print_file(xval, tenergy2, name="Energy")
    print_file(xval, mu, name="ChemPot")
    print_file(xval, mu1, name="ChemPot")

if __name__ == '__main__':

    main(sys.argv[1:])
