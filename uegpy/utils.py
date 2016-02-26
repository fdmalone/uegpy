'''Useful functions'''

import numpy as np
import scipy as sc


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
    Madelung contribution to the total energy. Please cite these guys and 
    L.M. Fraser et al. Phys. Rev. B 53, 1814 whose functional form they fitted
    to.

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

def add_mad(system, frame):
    ''' Add Madelung constant to data.

Parameters
----------
system : class
    system being studied.
frame : Pandas data frame
    Frame containing total energies.

Returns
-------
frame : Pandas data frame
    Frame with energies per particle including Madelung constant where
    appropriate.

'''

    names = system.ne

    for name in frame.columns:
        if U in name or V in name:
            frame[name] = frame[name]/system.ne + 0.5*madelung_approx(system)

    return frame
