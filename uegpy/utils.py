'''Useful functions'''

import numpy as np


def fermi_factor(ek, mu, beta):
    ''' Usual fermi factor.

Parameters
----------
ek : float
    Single particle eigenvalue.
mu : float
    Chemical potential.
beta : float
    Inverse temperature.

Returns
-------
f_k : float
    Fermi factor.

'''

    return 1.0/(np.exp(beta*(ek-mu))+1)


def fermi_block(ek, mu, beta):
    ''' Usual fermi factor blocking factor, i.e., fb_k = 1 - f_k.

Parameters
----------
ek : float
    Single particle eigenvalue.
mu : float
    Chemical potential.
beta : float
    Inverse temperature.

Returns
-------
fb_k : float
    Fermi blocking factor.

'''

    return 1.0 - fermi_factor(ek, mu, beta)


def madelung_approx(system):
    ''' Use expression in Schoof et al. (PhysRevLett.115.130402) for the
    Madelung contribution to the total energy.

Parameters
----------
system: class
    system being studied.

Returns
-------
v_M: float
    Madelung potential (in Hartrees).

'''

    return (-2.837297 * (3.0/(4.0*sc.pi))**(1.0/3.0) *
            system.ne**(-1.0/3.0) * system.rs**(-1.0))
