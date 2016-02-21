'''Evaluate properties of system with a finite particle number'''

def centre_of_mass_obsv(system, beta):
    ''' Calculate centre of mass partition function and total energy.

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
    ''' Calculate average number of electrons. 

Parameters
----------
mu : float 
    chemical potential. 
ne : int
    Number of electrons.
spval : array
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

    N = 0

    for speig in range(0,len(spval)):

        N = N + spval[speig][0]/(np.exp(-beta*(mu-spval[speig][1]))+1)

    return (2.0/pol)*N


def chem_pot_sum(system, spval, beta):
    ''' Find the chemical potential for finite system. 

Parameters
----------
system : class
    System class containing system information.
spval : array
    Single particle eigenvalues and degeneracies.
beta : float
    Inverse temperature.

Returns
-------

mu : float
   Chemical potential. 

'''
    return sc.optimize.fsolve(nav_sum, system.ef, args=(system.ne, evals, beta, system.pol))[0]

