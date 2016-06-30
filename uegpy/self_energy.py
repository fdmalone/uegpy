'''Module for evaluating Green's function for the electron gas'''

import numpy as np
import scipy as sc
import dielectric as di
import utils as ut
import greens_functions as gf

def angular_integrand(u, k, q, xi, beta, mu):

    omega = 0.5*k*k + 0.5*q*q + k*q*u

    return (
        di.im_chi_rpa(omega, q, beta, mu) *
        (ut.bose_factor(omega, beta)+ut.fermi_factor(omega, mu, beta))
    )

def q_integrand(q, k, xi, beta, mu):

    return (
        ut.vq(q) * sc.integrate.quad(angular_integrand, -1, 1,
                                  args=(k, q, xi, beta, mu))[0]
    )


def f_qu(q, k, u, xi, beta, mu):

        omega = 0.5*k*k + 0.5*q*q + k*q*u
        return (
           ut.vq(q) * (ut.bose_factor(omega, beta)+ut.fermi_factor(omega, mu, beta)) * di.im_chi_rpa(omega-xi, q, beta, mu)
        )


def im_sigma_rpa(k, xi, beta, mu):
    ''' Imaginary part of RPA self energy.

Parameters
----------
beta : float
    Inverse temperature
ek : float
    single particle energy
l : integer
    Index of Matsubara frequency

Returns
-------
g0 : float
    Matsubara Green's function.
'''


    I = sc.integrate.quad(q_integrand, 0.01, 3, args=(k, xi, beta, mu))[0]

    return (1.0 / sc.pi) * I


# def nk_rpa(beta, mu):

    # omega = np.
