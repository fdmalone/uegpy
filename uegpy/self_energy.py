'''Module for evaluating Green's function for the electron gas'''

import numpy as np
import scipy as sc
import dielectric as di
import utils as ut
import greens_functions as gf
import matplotlib.pyplot as pl
import pandas as pd

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

    omega = np.linspace(-omax, omax, nomega)
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
        re_eps[:, iq] = [1.0+di.kramers_kronig_eta(im_eps[:, iq], omega, o, do=omega[1]-omega[0]) for (io, o) in enumerate(omega)]
        denom = re_eps[:, iq]**2.0 + im_eps[:, iq]**2.0 + delta**2.0
        im_eps_inv[:, iq] = -im_eps[:, iq] / denom
        re_eps_inv[:, iq] = re_eps[:, iq] / denom

    return (re_eps_inv, im_eps_inv)

def angular_integral(q, im_eps_inv, xi, k, beta, mu, u_grid, omega_grid):

    # Values of E_{k+q} - xi
    omega_new = np.array([(0.5*k*k+0.5*q*q+k*q*u-mu)-xi for u in u_grid])
    # Integrand found by interpolating Im[1/eps] at new frequency values.
    F = [np.interp(o, omega_grid, im_eps_inv) * (ut.bose_factor(o, beta)+ut.fermi_factor(o+xi, 0, beta)) for o in omega_new]
    # Finally integrate F
    I = sc.integrate.simps(F, dx=(u_grid[1]-u_grid[0]))

    return I


def im_g0w0_self_energy(xi, k, beta, mu, im_eps_inv, nupoints, nkpoints, kmax, omega_grid):

    u_grid = np.linspace(-1, 1, nupoints)
    qp_grid = np.linspace(0, kmax, nkpoints)

    # Integrate along q direction.
    I = [angular_integral(q, im_eps_inv[:,idx], xi, k, beta, mu, u_grid, omega_grid) for (idx, q) in enumerate(qp_grid)]

    q_integral = sc.integrate.simps(I, dx=(qp_grid[1]-qp_grid[0]))

    return 1.0 / (sc.pi) * q_integral


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


def spectral_function(im_sigma, re_sigma, mu, k, omega_grid):

    numerator = abs(im_sigma)
    denominator = ((omega_grid+mu-0.5*k*k-re_sigma)**2.0+im_sigma**2.0)
    A = (1.0/sc.pi) * numerator / denominator

    return A


def momentum_distribution(A, beta, mu, k, omega_grid):

    n_k = sc.integrate.simps(A * ut.fermi_factor(0.5*k*k, mu, beta), dx=omega_grid[1]-omega_grid[0])

    return n_k


def hartree_fock(k, beta, mu, qmax):

    qvals = np.linspace(0, qmax, 2000)
    integrand = [ut.fermi_factor(0.5*q*q, mu, beta)*q*np.log(np.abs((k*k+q*q-2*k*q)/(k*k+q*q+2*k*q))) for q in qvals]

    return 1.0 / (2*sc.pi*k) * sc.integrate.simps(integrand, dx=qvals[1])


def write_table(table, row, column, name, variables, calc_type):

    df = pd.DataFrame()

    for (ik, k) in enumerate(column):

        df[k] = table[:, ik]

    f = write_header(name, variables, calc_type)
    df.colums = column
    df.set_index(row).to_csv(f)
    f.close()


def read_table(name):

    f = pd.read_csv(name, index_col=0)

    a = np.zeros(np.shape(f))

    for (ik, k) in enumerate(f.columns.values):

        a[:, ik] = f[k]

    cols = np.array(f.columns.values, dtype=float)
    rows = f.index.values

    return (a, rows, cols)


def write_header(filename, variables, calc_type='None'):

    f = open(filename, 'a')
    rev = ut.get_git_revision_hash()
    f.write('# -----------------\n')
    f.write('# Running uegpy at: %s\n'%rev)
    f.write('# -----------------\n')
    f.write('# System Details\n')
    f.write('# -----------------\n')
    for (key, value) in variables.items():
        f.write('# %s: %f\n'%(key, value))

    f.write('# -----------------\n')
    f.write('# Calculation type: %s\n'%calc_type)
    f.write('# -----------------\n')

    return f
