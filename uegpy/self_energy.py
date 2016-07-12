'''Module for evaluating Green's function for the electron gas'''

import numpy as np
import scipy as sc
import dielectric as di
import utils as ut
import greens_functions as gf
import matplotlib.pyplot as pl

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


def tabulate_dielectric(beta, mu, omax, kmax, nomega, nkpoints, zeta, delta=0.001):

    omega = np.linspace(0, omax, nomega)
    qvals = np.linspace(0, kmax, nkpoints)

    im_eps = np.zeros((len(omega), len(qvals)))
    re_eps = np.zeros((len(omega), len(qvals)))
    im_eps_inv = np.zeros((len(omega), len(qvals)))
    re_eps_inv = np.zeros((len(omega), len(qvals)))

    for (iq, q) in enumerate(qvals):
        if iq == 0:
            im_eps[:, iq] = np.zeros(len(omega))
        else:
            im_eps[:, iq] = np.array([-(2-zeta)*ut.vq(q)*di.im_lind_smeared(o, q, beta, mu, delta=delta, qmax=kmax)
                                                                    for o in omega])
        re_eps[:, iq] = [1.0+di.kramers_kronig(im_eps[:, iq], omega, o, io, do=omega[1]) for (io, o) in enumerate(omega)]
        denom = re_eps[:, iq]**2.0 + im_eps[:, iq]**2.0 + delta**2.0
        im_eps_inv[:, iq] = -im_eps[:, iq] / denom
        re_eps_inv[:, iq] = re_eps[:, iq] / denom

    return (re_eps_inv, im_eps_inv, re_eps, im_eps)



def im_g0w0_self_energy(xi, k, beta, mu, im_eps_inv, nupoints, nkpoints, kmax, omega_grid):

    def angular_integral(q, im_eps_inv, xi, k, beta, mu, u_grid, omega_grid):
        # Grid for angular integral
        # Values of E-E_{k-q}
        omega_new = np.array([xi-(0.5*k*k+0.5*q*q-k*q*u) for u in u_grid])
        # Integrand found by interpolating Im[1/eps] at new frequency values.
        #F = [np.interp(np.abs(o), omega_grid, im_eps_inv) * (1.0+ut.bose_factor(o, beta)-ut.fermi_factor(o-xi, mu, beta)) for o in omega_new]
        F = [np.interp(np.abs(o), omega_grid, im_eps_inv) * (ut.step(o, 0)-ut.step(mu, np.abs(o-xi))) for o in omega_new]
        # Integrand found by interpolating Im[1/eps] at new frequency values.
        # F = []
        # for o in omega_new:
            # if o > 0 and o < omega_grid[-1]:
                # F.append(np.interp(o, omega_grid, im_eps_inv) * (ut.step(o, 0)-ut.step(mu, np.abs(o-xi))))
            # elif o < 0 and -o < omega_grid[-1]:
                # F.append(-np.interp(np.abs(o), omega_grid, im_eps_inv) * (ut.step(o, 0)-ut.step(mu, np.abs(o-xi))))
            # else:
                # F.append(0)
        # Finally integrate F
        I = sc.integrate.simps(F, dx=(u_grid[1]-u_grid[0]))

        return (I, F)

    u_grid = np.linspace(-1, 1, nupoints)
    qp_grid = np.linspace(0, kmax, nkpoints)

    # Integrate along q direction.
    I = [angular_integral(q, im_eps_inv[:,idx], xi, k, beta, mu, u_grid, omega_grid)[0] for (idx, q) in enumerate(qp_grid)]

    q_integral = sc.integrate.simps(I, dx=(qp_grid[1]-qp_grid[0]))

    return - 1.0 / (sc.pi) * q_integral


def re_g0w0_self_energy(xi, k, beta, mu, im_eps_inv, nupoints, nkpoints, kmax, omega_grid):

    def frequency_integral(xi, k, beta, mu, im_eps_inv, e_kq, omega_grid, delta):

        integrand = [im_eps_inv[idx]*(1+ut.bose_factor(o, beta)-ut.fermi_factor(e_kq, mu, beta))/(xi-e_kq-o+delta) for (idx, o) in enumerate(omega_grid)]

        return 2 * di.kramers_kronig(integrand, omega_grid, xi, idx, omega_grid[1])

    def angular_integral(xi, k, beta, mu, im_eps_inv, q, u_grid, omega_grid, delta):

        integrand = [frequency_integral(xi, k, beta, mu, im_eps_inv, 0.5*k*k+0.5*q*q-k*q*u, omega_grid, delta) for u in u_grid]

        return sc.integrate.simps(integrand, dx=u_grid[2]-u_grid[1])


    u_grid = np.linspace(-1, 1, nupoints)
    qp_grid = np.linspace(0, kmax, nkpoints)

    I = [angular_integral(xi, k, beta, mu, im_eps_inv[:,idx], q, u_grid, omega_grid, delta=0.001) for (idx, q) in enumerate(qp_grid)]

    q_integral = sc.integrate.simps(I, dx=(qp_grid[1]-qp_grid[0]))

    return - 1.0 / (sc.pi) * q_integral


def g0w0_self_energy(xi, k, beta, mu, im_eps_inv, nupoints, nkpoints, kmax, omega_grid):

    im_sigma = im_g0w0_self_energy(xi, k, beta, mu, im_eps_inv, nupoints, nkpoints, kmax, omega_grid)

    re_sigma = di.kramers_kronig(im_sigma, omega_grid[:,kidx], xi+0.0001, idx, do=omega_grid[1])

    return (re_sigma, im_sigma)


