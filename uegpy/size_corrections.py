''' Utilities for finite size corrections. '''
import numpy as np
import scipy as sc
import structure as st
import utils as ut
import infinite as inf
import ueg_sys as ue


def conv_fac(nmax, alpha):
    '''Convergence factor for Hartree--Fock size corrections from Drummond et al.
    Warning: Exceptionally slow.

Parameters
----------
nmax : int
    Maximum kvector to consider.
alpha : float
    alpha parameter

Returns
-------
C_HF : float
    Convergence constant.
'''

    g = 0
    old = 0

    for ni in range(-nmax,nmax+1):
        for nk in range(-nmax,nmax+1):
            for nj in range(-nmax,nmax+1):
                n = ni**2 + nj**2 + nk**2
                old = g
                if n != 0:
                    g += np.exp(-alpha*n) / np.sqrt(n)
                diff = g - old

    return 2.0*sc.pi/alpha-g


def fxc_correction(rs, theta, zeta, N):
    '''Free energy correction using the BCDC finite size correction for V

Parameters
----------
rs : float
    Wigner-Seitz parameter.
theta : float
    Degeneracy temperature.
zeta : int
    Spin polarisation.
N : int
    Number of electrons.

Returns
-------
delta_f_xc(V) : float
    RPA finite size correction.
'''

    omega_p = ut.plasma_freq(rs)

    def integrand(x, theta, zeta):

        ef = ut.ef(x, zeta)

        return x**(-0.5) * 1.0/(np.tanh((0.5/(theta*ef))*(3.0/x)**(3/2.)))

    return (
        (0.25 * 3**0.5 / (rs**2.0*N) *
        sc.integrate.quad(integrand, 0, rs, args=(theta, zeta))[0])
    )


def bcdc(rs, theta, zeta, N):
    '''Potential energy correction using the BCDC finite size correction for V.

    Ref: Ethat W. Brown et al. Phys. Rev. Lett. 110, 146405 (2013).

Parameters
----------
rs : float
    Wigner-Seitz parameter.
theta : float
    Degeneracy temperature.
zeta : int
    Spin polarisation.
N : int
    Number of electrons.

Returns
-------
delta_V : float
    RPA finite size correction.
'''

    omega_p = ut.plasma_freq(rs)
    beta = 1.0 / (ut.ef(rs, zeta)*theta)

    return 0.25 * omega_p / (N * np.tanh(0.5*beta*omega_p))


def sk_integral(rs, theta, eta, zeta, kf, kmin, kmax, lmax):
    '''Integral over RPA structure factor.

Parameters
----------
rs : float
    Wigner-Seitz parameter.
theta : float
    Degeneracy temperature.
zeta : int
    Spin polarisation.
kf : float
    Fermi wave vector.
kmin: float
    Lower limit.
kmax: float
    Upper limit.
lmax : int
    Upper limit on Matsubara sum.

Returns
-------
intg : float
    Integral over s_q.
'''

    def integrand(q, rs, theta, eta, zeta, kf, lmax):

        return (
            st.rpa_matsubara(q/kf, rs, theta, eta, zeta, lmax)
        )

    return (
        1.0/(sc.pi) * (sc.integrate.quad(integrand, kmin, kmax, args=(rs, theta,
                          eta, zeta, kf, lmax)))[0]
    )


def v_integral(rs, theta, eta, zeta, kf, kmin, kmax, lmax):
    '''Potential energy evaluated from RPA structure factor.

    Useful to evaluate size corrections.


Parameters
----------
rs : float
    Wigner-Seitz parameter.
theta : float
    Degeneracy temperature.
zeta : int
    Spin polarisation.
kf : float
    Fermi wave vector.
kmin: float
    Lower limit.
kmax: float
    Upper limit.
lmax : int
    Upper limit on Matsubara sum.

Returns
-------
V : float
    RPA potential energy.
'''

    def integrand(q, rs, theta, eta, zeta, kf, lmax):

        return (
            st.rpa_matsubara(q, rs, theta, eta, zeta, lmax) - 1.0
        )

    return (
        kf / (sc.pi) * (sc.integrate.quad(integrand, kmin, kmax, args=(rs, theta,
                                         eta, zeta, kf, lmax)))[0]
    )


def mad_integral(theta, eta, zeta, kf, kmin, kmax, nmax):
    '''Evaluate integral of uniform distribution of point charges.

Parameters
----------
rs : float
    Wigner-Seitz parameter.
theta : float
    Degeneracy temperature.
zeta : int
    Spin polarisation.
kf : float
    Fermi wave vector.
kmin: float
    Lower limit.
kmax: float
    Upper limit.
lmax : int
    Upper limit on Matsubara sum.

Returns
-------
v_s: float
   Potential energy from static distribution of point charges.
'''
    def integrand(q, theta, eta, zeta, kf, nmax):

        return (
            -1
        )

    return (
        1.0/(sc.pi) * (sc.integrate.quad(integrand, kmin, kmax, args=(theta,
                                         eta, zeta, kf, nmax)))[0]
    )


def v_summation(rs, theta, eta, zeta, kf, lmax, qvals, L):
    '''Potential energy evaluated at discrete kpoints.

Parameters
----------
rs : float
    Wigner-Seitz parameter.
theta : float
    Degeneracy temperature.
zeta : int
    Spin polarisation.
kf : float
    Fermi wave vector.
lmax : int
    Upper limit on Matsubara sum.
qvals: list
    magnitude of kpoints to evaluate structure factor at.
L : float
    Box length.

Returns
-------
V: float
    RPA potential energy evalutes in a finite box (assuming the structure factor
            is converged).
'''
    v = sum([ut.vq(q)*(st.rpa_matsubara(q/kf, rs, theta, eta, zeta, lmax)
              -1.0) for q in qvals])

    return 1.0 / (2.0*L**3.0) * v


def mad_summation(qvals, L):
    '''Evaluate sum of uniform distribution of point charges.

Parameters
----------
qvals: list
    magnitude of kpoints to evaluate structure factor at.
L : float
    Box length.

Returns
-------
v_s : float
    Contribution to potential energy from periodically repeated point charges.
'''
    v = sum([ut.vq(q)*(-1.0) for q in qvals])

    return 1.0 / (2.0*L**3.0) * v


def fxc_correction_quad(rs, dv):
    '''Evalute correction to exchange-correlation free energy.

Parameters
----------
rs : list
    Grid of densities.
dv : list
    Corresponding grid of potential energy size corrections.

Returns
-------
df_xc : float
    Size correction to f_xc.
'''

    return rs[-1]**(-2.0) * sc.integrate.simps(rs*dv, rs)
