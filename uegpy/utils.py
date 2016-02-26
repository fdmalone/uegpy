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


def magic_numbers(system):
    ''' Work out possible magic electron numbers (i.e. those that will fit in
    a closed shell) below nmax

Parameters
----------
system : class
    System being studied

Returns
-------
pol : list
    Polarised magic numbers. Unpolarised = 2 * pol
'''

    ne = 0
    pol = []

    for i in system.deg_e:
        ne = ne + i[0]
        pol.append(ne)

    return pol


def kinetic_cutoff(ne, theta):
    '''Determine number of planewaves necessary to converge kinetic energy.

    The kinetic energy converges exponentially once

    .. math::
        \varepsilon_c \approx kT

    The following extrapolates from a smaller system size to determine the cutoff
    necessary at any Theta, N value as

    e_c(N,Theta) =  alpha e_c(19,8) (Theta/8) (N/19)^{2/3}

Parameters
----------
ne : int
    Number of electrons
theta : float
    Reduced temperature.
Returns
-------
e_c : float
    Cutoff required (units of (2pi/L)^2
'''

    # Determined numerically using `ref`tests/find_max_m.py, ensures results are
    # converged within 1e-6 Ha.
    ec_ref = 98
    t_ref = 8

    sgn = np.sign(t_ref-theta)

    # Prefactor is emperically determined so this number will over estimate.
    return 1.15 * (theta/t_ref)**(sgn) * ec_ref * (ne/19.0)**(2./3.)


def kinetic_plane_waves(ne, theta):
    '''Determine number of planewaves necessary to converge kinetic energy.

    The kinetic energy converges exponentially once

    .. math::
        \varepsilon_c \approx kT
        M \approx N \Theta^{3/2}

    The following extrapolates from a smaller system size to determine the cutoff
    necessary at any Theta, N value as

    M(N,Theta) = (theta/8)^{3/2} M(19,8) (N/19)

Parameters
----------
ne : int
    Number of electrons
theta : float
    Reduced temperature.
Returns
-------
M : float
    Number of planewaves required
'''

    # Determined numerically using `ref`tests/find_max_m.py, ensures results are
    # converged within 1e-6 Ha.
    M_ref = 11459
    t_ref = 8

    sgn = np.sign(t_ref-theta)

    # Prefactor is emperically determined so this number will overestimate.
    return 1.23 * (theta/t_ref)**(1.5*sgn) * M_ref * (ne/19.0)
