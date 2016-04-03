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


def rpa_structure_factor(q, beta, mu):

    return (
        -1.0/(2.0*sc.pi) * sc.integrate.quad(s_integrand, 0, np.inf,
                                                args=(q, beta, mu))[0]
    )

def s_integrand(omega, q, beta, mu):

    return im_chi_rpa(omega, q, beta, mu) * 1.0/(np.tanh(0.5*beta*omega))


def re_lind0(q, omega, kf):

    qb = q / kf
    nu_pl = omega/(q*kf) + q/(2*kf)
    nu_mi = omega/(q*kf) - q/(2*kf)

    return (
        (kf/(2*sc.pi**2))*(0.5
            - (1-nu_mi**2.0)/(4*qb)*np.log(np.abs((nu_mi+1)/(nu_mi-1)))
            + (1-nu_pl**2.0)/(4*qb)*np.log(np.abs((nu_pl+1)/(nu_pl-1))))
    )


def im_lind0(q, omega, kf):

    qb = q / kf
    nu_pl = omega/(q*kf) + q/(2*kf)
    nu_mi = omega/(q*kf) - q/(2*kf)

    return (
        ((0.5*kf**2.0)/(4*sc.pi*q))*(step_ab(1, nu_mi**2.0)*(1-nu_mi**2.0) -
                                        step_ab(1, nu_pl**2.0)*(1-nu_pl**2.0))
    )


def step_ab(x,a):
    '''Theta(x-a)'''

    if a <= x:
        return 1
    else:
        return 0

def im_chi_rpa(omega, q, beta, mu):
    '''Imaginary part of rpa density-density response function.

Parameters
----------
omega : float
    frequency
q : float
    (modulus) of wavevector considered.
beta : float
    Inverse temperature.
mu : float
    Chemical potential.

Returns
-------
chi_rpa : float
    Imaginary part of RPA density-density response function.

'''

    vq = 4.0*sc.pi / q**2.0
    num = im_lind(beta, mu, q, omega)**2.0
    denom = ((1.0-vq*re_lind(beta, mu, q, omega))**2.0 +
                                          (vq*im_lind(beta, mu, q, omega))**2.0)

    return num / denom


def re_lind(beta, mu, q, omega):
    '''Real part of free-electron Lindhard density-density response function.

Parameters
----------
beta : float
    Inverse temperature.
mu : float
    Chemical potential.
q : float
    (modulus) of wavevector considered.
omega : float
    frequency

Returns
-------
re_chi : float
    Imaginary part of thermal Lindard function.

'''


    return (
        (1.0/((2*sc.pi)**2.0*q)) * sc.integrate.quad(re_lind_integrand, 0, np.inf,
                                                   args=(beta, mu, q, omega))[0]
    )


def re_lind_integrand(k, beta, mu, q, omega):
    ''' Integrand for real part of Lindhard function

Parameters
----------
k : float
    integration variable (modulus of wavevector).
beta : float
    Inverse temperature.
mu : float
    Chemical potential.
q : float
    (modulus) of wavevector considered.
omega : float
    frequency

Returns
-------
int : float
    Integrand.
'''

    k_pl = (0.5*q**2.0 + omega) / q
    k_mi = (0.5*q**2.0 - omega) / q
    #print ((np.log(np.abs((k_pl+k)/(k_pl-k)))
                                             ##- np.log(np.abs((k_mi+k)/(k_mi-k)))))


    return (
        k * fermi_factor(0.5*k**2.0, mu, beta) * (np.log(np.abs((k_pl+k)/(k_pl-k)))
                                             + np.log(np.abs((k_mi+k)/(k_mi-k))))
    )


def im_lind(beta, mu, q, omega):
    '''Imaginary part of free-electron Lindhard density-density response function.

Parameters
----------
beta : float
    Inverse temperature.
mu : float
    Chemical potential.
q : float
    (modulus) of wavevector considered.
omega : float
    frequency

Returns
-------
im_chi : float
    Imaginary part of thermal Lindard function.

'''


    eq = 0.5*q**2.0
    e_pl = (eq + omega)**2.0/(4*eq)
    e_mi = (eq - omega)**2.0/(4*eq)

    return (
        1.0/(4*sc.pi*beta*q) * np.log((1.0+np.exp(-beta*(e_mi-mu)))/
                                         (1.0+np.exp(-beta*(e_pl-mu))))
    )
