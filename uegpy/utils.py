'''Useful functions.'''

import numpy as np
import scipy as sc
from scipy import integrate
import subprocess
import sys


def fermi_factor(ek, mu, beta):
    ''' Usual fermi factor:
    .. math::
        f_k = \\frac{1}{e^{\\beta(\\varepsilon_k-\\mu)}+1}

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
    ''' Usual fermi factor blocking factor, i.e., :math:`\\bar{f}_k = 1-f_k`.

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


def fermi_angle(u, k, q, mu, beta):
    ''' Fermi factor for :math:`k + q`. Used for integration.

Parameters
----------
u : float
    Integration variable = :math:`\cos \theta`.
k, q : float
   Magnitude of wavevectors
mu : float
    Chemical potential.
beta : float
    Inverse temperature.

Returns
-------
f_kq : float
    Fermi factor.

'''

    return fermi_factor(0.5*(k**2.0+2.0*k*q*u+q**2.0), mu, beta)


def madelung_approx(rs, ne):
    ''' Use expression in Schoof et al. (PhysRevLett.115.130402) for the
    Madelung contribution to the total energy. Please cite these guys and
    L.M. Fraser et al. Phys. Rev. B 53, 1814 whose functional form they fitted
    to.

Parameters
----------
rs : float
    Wigner-Seitz radius.
ne : int
    Number of electrons.

Returns
-------
v_M: float
    Madelung potential (in Hartrees).

'''

    return (-2.837297*(3.0/(4.0*sc.pi))**(1.0/3.0)*ne**(-1.0/3.0)*rs**(-1.0))


def add_mad(system, frame):
    ''' Add Madelung constant to data.

Parameters
----------
system : class
    system being studied.
frame : :class:`pandas.DataFrame`
    Frame containing total energies.

Returns
-------
frame : :class:`pandas.DataFrame`
    Frame with energies per particle including Madelung constant where
    appropriate.

'''

    mads = [x for x in frame.columns if 'u' in x or 'v' in x]
    mads = [x for x in mads if 'error' not in x]
    rest = [x for x in frame.columns if x not in mads and 'ne' not in x and
            'Beta' not in x and 'M' not in x and 'rs' not in x]

    for name in mads:
        frame[name] = frame[name]/system.ne + 0.5*madelung_approx(system)
    for name in rest:
        frame[name] = frame[name] / system.ne

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
        \\varepsilon_c \\approx kT

    The following extrapolates from a smaller system size to determine the cutoff
    necessary at any :math:`\\Theta, N` value as

    .. math::
        \\varepsilon_c(N,\\Theta) =  \\alpha \\varepsilon_c(19,8) (\\Theta/8)
                                      (N/19)^{2/3}

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
    ec_ref = 207
    t_ref = 8

    # Prefactor is emperically determined so this number will over estimate.
    return 1.15 * (theta/t_ref) * ec_ref * (ne/19.0)**(2./3.)


def kinetic_plane_waves(ne, theta):
    '''Determine number of planewaves necessary to converge kinetic energy.

    The kinetic energy converges exponentially once

    .. math::
        M \\approx N \\Theta^{3/2}

    The following extrapolates from a smaller system size to determine the cutoff
    necessary at any :math:`\\Theta, N` value as

    .. math::
        M(N,\\Theta) = (\\Theta/8)^{3/2} M(19,8) (N/19)

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


def get_git_revision_hash():
    '''Return git revision.

    Adapted from: http://stackoverflow.com/questions/14989858/get-the-current-git-hash-in-a-python-script

Returns
-------
sha1 : string
    git hash with -dirty appended if uncommitted changes.
'''

    src = sys.path[0]

    sha1 = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                   cwd=src).strip()
    suffix = subprocess.check_output(['git', 'status', '--porcelain'],
                                     cwd=src).strip()
    if suffix:
        return sha1 + '-dirty'
    else:
        return sha1


def add_frame(f1, f2, val1, op='+', val2=None, label=None, err=False):
    '''Add or subtract two frames and take care of errorbars.

Parameters
----------
f1,f2 : :class:`pandas.DataFrame`
     Frames to add/subtract.
Returns
-------
f : :class:`pandas.DataFrame`
    f1 +/- f2
'''

    if val2 == None:
        val2 = val1
    if label == None:
        label = val1

    if op == '+':
     f1[label] = f1[val1] + f2[val2]
    else:
     f1[label] = f1[val1] - f2[val2]

    if err:
        f1[label+'_error'] = (
                np.sqrt(f1[val1+'_error']**2.0+f2[val2+'_error']**2.0)
        )

    return f1


def vq(q):
    ''' Coulomb interaction.

Parameters
----------
q : float.
    Magnitude of

Returns
-------
vq : float
    Coulomb interaction.

'''

    return 4.0*sc.pi / (q*q)


def vq_vec(q):
    ''' Coulomb interaction.

Parameters
----------
q : vector.
    Magnitude of

Returns
-------
vq : float
    Coulomb interaction.

'''

    return 4.0*sc.pi / np.dot(q,q)


def ef(rs, zeta):
    ''' Fermi Energy for 3D UEG.

Parameters
----------
rs : float
    Density parameter.
zeta : int
    Spin polarisation.

Returns
-------
ef : float
    Fermi Energy.

'''

    return 0.5 * (9.0*(zeta+1)*sc.pi*0.25)**(2.0/3.0) * rs**(-2.0)


def step_angle(u, k, q, kf):
    ''' Heaviside steb function for k+q.

Parameters
----------
a, b : float
    Arguments of step function.

Returns:
--------
theta(a-b): float
    Heaviside step function.
'''

    if (np.sqrt(k**2.0+q**2.0+2*k*q*u) > kf):
        return 0
    else:
        return 1


def step(a, b):
    ''' Heaviside steb function i.e., :math:`\theta(a-b)`

Parameters
----------
a, b : float
    Arguments of step function.

Returns:
--------
theta(a-b): float
    Heaviside step function.
'''

    if (a < b):
        return 0
    else:
        return 1


def step_angle(u, k, q, kf):
    ''' Heaviside steb function for k+q.

Parameters
----------
a, b : float
    Arguments of step function.

Returns:
--------
theta(a-b): float
    Heaviside step function.
'''

    if (np.sqrt(k**2.0+q**2.0+2*k*q*u) > kf):
        return 0
    else:
        return 1


def plasma_freq(rs):
    ''' Plasma frequency for 3D UEG.

Parameters
----------
rs : float
    Density parameter.

Returns
-------
omega_p : float
    Plasma frequency.

'''

    return (3.0/rs**3.0)**0.5


def alpha(zeta):
    ''' Alpha Parameter for ueg.

Parameters
----------
zeta : int

Returns
-------
alpha : float

'''

    return (4.0/(9*sc.pi*(zeta+1)))**(1.0/3.0)


def gamma(rs, theta, zeta):
    ''' Classical plasma coupling parameter for 3D UEG

Parameters
----------
rs : float
    Density Parameter.
theta : float
    Degeneracy temperature.
zeta : int
    Spin polarisation.

Returns
-------
gamma : float
    Coupling parameter.

'''

    return 2.0 * alpha(zeta)**2.0 * rs / theta


def rs_gamma(gamma, theta, zeta):
    ''' Find rs give a gamma and theta.

Parameters
----------
rs : float
    Density Parameter.
theta : float
    Degeneracy temperature.
zeta : int
    Spin polarisation.

Returns
-------
gamma : float
    Coupling parameter.

'''

    return gamma * theta / (2.0 * alpha(zeta)**2.0)

def theta_gamma(gamma, rs, zeta):
    ''' Find theta give a gamma and ts.

Parameters
----------
rs : float
    Density Parameter.
theta : float
    Degeneracy temperature.
zeta : int
    Spin polarisation.

Returns
-------
gamma : float
    Coupling parameter.

'''

    return 2.0 * alpha(zeta)**2.0 * rs / gamma
