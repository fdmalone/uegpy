'''Finite size corrections at various levels of approximation'''

import numpy as np
import scipy as sc
from utils import fermi_factor

def hf_structure_factor(q, rs, beta, mu, zeta):
    '''Static structure factor at Hartree--Fock level:

    .. math::
        S(q) = 1 - \\frac{3}{2 (2\pi r_s)^3}
               \int_0^{\infty} dk k^2 f_k \int_{-1}^{1} du
               \frac{1}{e^{\beta(0.5*(k^2+q^2+2kqu)}+1}

Parameters
----------
q : float
    kvalue to calculate structure factor at.
rs : float
    Wigner-Seitz radius.
beta : float
    Inverse temperature.
mu : float
    Chemical potential.
zeta : int
    Spin polarisation

Returns
-------
S(q) : float
    Static structure factor.

'''


    return (
        0.5*(1.0 - (rs**3.0/(3.0*sc.pi)) *
               sc.integrate.quad(k_integrand, 0, np.inf, args=(q, mu, beta))[0])
    )

def k_integrand(k, q, mu, beta):

    return k**2.0 * fermi_factor(0.5*k**2.0, mu, beta) * sc.integrate.quad(f_kq, -1.0, 1.0, args=(k, q, mu, beta))[0]

def f_kq(u, k, q, mu, beta):

    return fermi_factor(0.5*(k**2.0+2.0*k*q*u+q**2.0), mu, beta)

def ground_state_integral(q, rs, kf):

    return 0.5*(1 - (rs**3.0/(3.0*sc.pi)) *
                    sc.integrate.quad(k0_int, 0, 2*kf, args=(q, kf))[0])

def k0_int(k, q, kf):

    return (k**2.0 *
           step(k, kf) *
           sc.integrate.quad(step_1, -1, 1, args=(k, q, kf))[0])

def step_1(u, k, q, kf):

    if (np.sqrt(k**2.0+q**2.0+2*k*q*u) > kf):
        return 0
    else:
        return 1


def step(k, kf):

    if (k > kf):
        return 0
    else:
        return 1


def ground_state(q, kf):

    if (q <= 2*kf):
        return 0.5*(3.0/4.0 * q/kf - 1.0/16.0 * (q/kf)**3.0)
    else:
        return 0.5

def conv_fac(nmax, alpha):

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

    return (2.0*sc.pi/alpha-g, g, diff)
