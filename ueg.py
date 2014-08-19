#!/usr/bin/env python

import sys
import scipy as sc
import numpy as np
import matplotlib.pyplot as pl

def kf(ne, L):

    k_f = (6 * sc.pi**2 * ne/L**3)**(1/3.0)

    return k_f

def ef(ne, L):

    e_f = (3*sc.pi**2*ne/L**3)**(2/3.)

    return e_f

def setup(rs, ne):

    # System Lenght.
    L = rs*(4*ne*sc.pi/3.)**(1/3.)
    # k = 2*pi/L n
    kfac = 2*sc.pi/L

    return (L, kfac)

def sp_energies(kfac, ecut):

    nmax = int((ecut/kfac)**(1/2.))

    print "nmax: ", nmax
    spval = []

    for ni in range(-nmax, nmax+1):
        for nj in range(-nmax, nmax+1):
            for nk in range(-nmax, nmax+1):
                spe = 0.5*kfac**2*(ni**2 + nj**2 + nk**2)
                if (spe < ecut):
                    spval.append(spe)

    return spval

def dis_fermi(spval, ne):

    return 0.5*(spval[ne-1]+spval[ne])
    #return spval[ne-1]

def nav_deriv(spval, x, beta):

    deriv = 0

    for speig in spval:

        deriv = deriv + beta*np.exp(-beta*(x - speig))/(np.exp(-beta*(x - speig))+1)

    return deriv

def nav(spval, ne, beta, guess):

    mu = guess
    N = 0
    it = 0
    n_sum = []

    for speig in spval:

        N = N + 1/(np.exp(-beta*(mu - speig))+1)
        n_sum.append(N)

    return N
    #pl.plot(n_sum, marker='o', color='blue', markersize=2, linewidth=2)
    #pl.xlabel("# eigenvalues")
    #pl.ylabel(r"$\langle \hat{N} \rangle$")
    #pl.show()

def newton(spval, ne, beta, efermi, de):

    mu = efermi
    it = 0
    mu_new = 0
    npart = 0

    while (abs(ne-npart) > de):
        #print nav(spval, ne, beta, mu), nav_deriv(spval, mu, beta), (nav(spval, ne, beta, mu)-ne)/nav_deriv(spval, mu, beta)
        mu_new = mu - (nav(spval, ne, beta, mu)-ne)/(nav_deriv(spval, mu, beta))
        #print npart
        npart = nav(spval, ne, beta, mu_new)
        mu = mu_new
        it = it + 1

    return mu

def chem_pot(spval, ne, efermi, de):

    beta = 0
    T = 0

    mu = []
    xval = []
    npart = []

    for x in range(1,29):
        beta = beta + 0.2
        mu.append(newton(spval, ne, beta, efermi, de))
        xval.append(beta)

    return(xval, mu)

def energy(beta, mu, spval):

    tot_energy = []
    tenergy = 0

    for temp in range(0,len(beta)):
        for speig in spval:
            tenergy = tenergy + speig/(np.exp(-beta[temp]*(mu[temp] - speig))+1)

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


def main(args):

    rs = float(args[0])
    ne = float(args[1])
    ecut = float(args[2])

    (L, kfac) = setup(rs, ne)
    # Fermi Momentum.
    k_f = kf(ne, L)
    # Fermi energy.
    e_f = ef(ne, L)
    # Single particle energies.
    spval = sp_energies(kfac, ecut)
    spval = np.array(spval)
    spval.sort()

    print "# of electrons: ", ne
    print "rs: ", rs
    print "System Length: ", L
    print "Fermi wavevector for infinite system: ", k_f
    print "Fermi energy for infinite system: ", e_f
    print "Fermi energy for discrete system: ", dis_fermi(spval, ne)
    print "kspace grid spacing: ", kfac
    #print "Single particle energies"
    #for i in spval:
        #print i

    #beta = 1.5
    #chem_pot = newton(spval, ne, beta, dis_fermi(spval,ne), 1e-8)
    #print "Chemical potential: ", chem_pot, nav(spval, ne, beta, chem_pot)
    (xval, mu) = chem_pot(spval, ne, dis_fermi(spval, ne), 1e-3)
    tenergy = [i for i in energy(xval, mu, spval)]
    spec_heat = specific_heat(xval, mu, spval)
    pl.subplot(3,1,1)
    pl.plot(xval, mu)
    pl.subplot(3,1,2)
    pl.plot(xval, tenergy)
    pl.subplot(3,1,3)
    pl.plot(xval, spec_heat)
    pl.show()

if __name__ == '__main__':

    main(sys.argv[1:])
