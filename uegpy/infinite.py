'''Evaluate properties of electron gas in thermodynamic limit using grand
   canonical ensemble'''
import numpy as np
import math
import scipy as sc
from scipy import integrate
import random as rand
from scipy import optimize
import utils as ut


def nav(beta, mu, zeta):
    '''Average density i.e. :math:`N/V`.

Parameters
----------
beta : float
    Inverse temperature.
mu : float
    Chemical potential.
zeta : int
    Spin polarisation.

Returns
-------
rho : float
    Average density.

'''

    return (
        (2-zeta) * (2.0**0.5/(2.0*sc.pi**2.0)) * beta**(-1.5)
                 * fermi_integral(0.5, mu*beta)
    )


def chem_pot(rs, beta, ef, zeta, method=nav, it=1):
    '''Find the chemical potential for infinite system.

Parameters
----------
rs : float
    Wigner-Seitz radius.
beta : float
    Inverse temperature.
ef : float
    Fermi energy.
zeta : int
    Spin polarisation.

Returns
-------
mu : float
   Chemical potential.

'''

    return (
        sc.optimize.fsolve(nav_diff, ef, args=(beta, rs, zeta, method, it))[0]
    )



def nav_diff(mu, beta, rs, zeta, method=nav, it=1):
    '''Deviation of average density for given :math:`\\mu` from true value.

Parameters
----------
mu : float
    Chemical potential.
beta : float
    Inverse temperature.
rho : float
    True density.
zeta : int
    Spin polarisation.

Returns
-------
dev : float
    :math:`n(\\mu) - n`.

'''

    rho = ((4*sc.pi*rs**3.0)/3.0)**(-1.0) # System density.
    return method(beta, mu, zeta) - rho


def fermi_integrand(x, nu, eta):
    '''Integrand of standard Fermi integral I(eta, nu), where:

    .. math::
        I(\\eta, \\nu) = \\int_0^{\\infty} \\frac{x^{\\nu}}{(e^{x-\\eta}+1)} dx

Parameters
----------
x : float
    integration variable.
nu : float
    Order of integral.
eta : float
    :math:`\\beta\\mu`.

'''

    return x**nu / (np.exp(x-eta)+1)


def fermi_integrand_deriv(x, nu, eta):
    ''' Derivative of integrand of standard Fermi integral :math:`I(eta, nu)`
    wrt beta.

    TODO : check this.

Parameters
----------
x : float
    integration variable.
nu : float
    Order of integral.
eta : float
    :math:`\\beta\\mu`.

'''

    return x**nu / (np.exp(x-eta)+2+np.exp(eta-x))


def fermi_integral(nu, eta):
    ''' Standard Fermi integral :math:`I(\\eta, \\nu)`, where:

    .. math::
        I(\\eta, \\nu) = \\int_0^{\\infty} \\frac{x^{\\nu}}{(e^{x-\\eta}+1)} dx

Parameters
----------
eta : float
    :math:`\\beta\\mu`.
nu : float
    Order of integral.

Returns
-------
:math:`I(\\eta, \\nu)` : float
    Fermi integral.

'''

    return sc.integrate.quad(fermi_integrand, 0, np.inf, args=(nu, eta))[0]


def energy_integral(beta, mu, rs, zeta):
    ''' Total energy at inverse temperature beta:

    .. math::
        U = (2-\\zeta) \\frac{2\\sqrt{2}}{3\\pi}r_s^3\\beta^{-5/2} I(3/2, \\eta)

Parameters
----------
eta : float
    beta * mu, for beta = 1 / T and mu the chemical potential.
nu : float
    Order of integral.

Returns
-------
I(eta, nu) : float
    Fermi integral.

'''

    return (
            (2-zeta) * (2**1.5/(3.0*sc.pi)) * rs**3.0 * beta**(-2.5)
                    * fermi_integral(1.5, mu*beta)
    )


def gc_free_energy_integral(beta, mu, rs):
    ''' Free energy:

    .. math::
        \Omega = 

Parameters
----------
beta : float
    Inverse temperature.
mu : float
    Chemical potential.
rs : float
    Wigner-Seitz radius.

Returns
-------
Omega : float
    Grand potential (Helmholtz Free energy.)

'''

    return (
            -(2/3.) * ((4*sc.pi*rs**3.0)/3.0) * np.sqrt(2.0)/sc.pi**2.0
                        * beta**(-5./2.) * fermi_integral(5./2., mu*beta)
    )


def hfx_integrand(eta, power=2.0):
    ''' Integrand of first order exchange contribution to internal energy.

    .. math::
        U_x = \int_0^{\infty} \frac{x^{\nu}}{(e^{x-nu}+1)} dx

    Todo : maths + reference.

Parameters
----------
eta : float
    beta * mu, for beta = 1 / T and mu the chemical potential.

Returns
-------
I(-1/2, nu)^2 : float
    Fermi integral.

'''

    return fermi_integral(-0.5, eta)**power


def hfx_integral(rs, beta, mu, zeta):
    ''' First-order exchange contribution to internal energy:

    .. math::
        \Omega =

Parameters
----------
rs : float
    Wigner-Seitz radius.
beta : float
    Inverse temperature.
mu : float
    Chemical potential.

Returns
-------
hfx : float
    Grand potential (Helmholtz Free energy.)

'''

    hfx = sc.integrate.quad(hfx_integrand, -np.inf, beta*mu)[0]

    return - (2-zeta) * rs**3.0/(3*sc.pi**2.0*beta**2.0) * hfx


def inversion_correction(rs, beta, mu, zeta):
    '''Correction to Helmholtz free energy when moving from grand canonical to
    canonical ensemble.

    Turns out to be:

    .. math::

        \\frac{1}{2\\sqrt{2}\\pi^4}\\beta^{-3/2}I_{-1/2}(\\eta_0)^{3}

Parameters
----------
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
corr : float
    Correction term for inversion process.

'''

    return 2**(-0.5)/(2.0*sc.pi**4.0)*beta**(-3./2.)*hfx_integrand(beta*mu, 3.0)


def thf_integrand_0(q, eq, k, beta, mu):

    # Fix fermi factor here.
    return (
        q * ut.fermi_factor(eq, mu, beta) * np.log(np.abs((k+q)/(k-q)))
    )


def thf_spect(k, beta, mu, n):

    # Units
    return 0.5*k**2.0 - thf_ex_rec(k, beta, mu, n)

def thf_ex_rec(k, beta, mu, n):

    if n == 1:
        if k == 0:
            return 1
        else:
            return k
    else:
        0.5*k**2.0 - ((1.0/(sc.pi*k))*sc.integrate.quad(thf_integrand_0, 0,
                   np.inf, args=(thf_ex_rec(k, beta, mu, n-1), k, beta, mu)))


def t0_hf_integrand(p, k):
    '''Integrand for T=0 hartree fock exchange potential.'''

    return p*(np.log(np.abs((k+p)/(k-p))))


def t0_spect(k, k_f):
    '''T=0 hf spectrum calculated using integral.'''

    return 0.5*k**2.0 - ((1.0/(sc.pi*k))*sc.integrate.quad(t0_hf_integrand, 0,
                                                   k_f, args=(k))[0])


def t0_spect_dis(k, k_f, dt=1e-6):
    '''T=0 HF spectrum calculated using discretised (explicitly).'''

    if k < k_f:
        q0 = np.arange(0, k-dt, dt)
        q1 = np.arange(k+dt, k_f, dt)
        q = np.concatenate((q0, q1), axis=0)
    else:
        q = np.arange(0, k_f, dt)
    return 1.0/(sc.pi*k) * sc.integrate.simps(t0_hf_integrand(q,k), q)


def nav_hf_integrand(k, beta, mu, n):

    return k**2.0*ut.fermi_factor(thf_spect(k, beta, mu, n), mu, beta)


def nav_hf(beta, mu, zeta, n):

    return (
        (2-zeta) * (2**0.5/(2.0*sc.pi**2.0))*sc.integrate.quad(nav_hf_integrand,
                                        0, np.inf, args=(beta, mu, n))[0]
    )


def sigmax_discrete(kv, eq, beta, mu, zeta, kmax, dq=0.1):

    sigma = np.zeros(len(eq))
    q = np.arange(0.1, kmax, dq)
    q_loc = []
    eq_loc = [] 
    it = 0
    # Look at interpolating.
    for k in kv:
        loc = 0
        for x in q:
            if k == x:
                q_loc = np.delete(q, loc)
                eq_loc = np.delete(eq, loc)
            loc += 1
        #print len(q_loc), len(eq_loc), len(eq)
        sigma[it] = (
            - 1.0/(sc.pi*k)*sc.integrate.simps(thf_integrand_0(q_loc,
                                                       eq_loc, k, beta, mu), q_loc)
        )
        it += 1
    return sigma


def self_consist(rs, beta, mu, zeta, ef):

    # Free electrons is our reference point.
    mu_old = chem_pot(rs, beta, ef, zeta)

    it = 1
    max_it = 2
    while it < max_it:
        # Calculate average particle number with given chemical potential and
        # spectrum.
        #nav_new = nav_hf(beta, mu_old, it)
        # Work out new chemical potential.
        #mu = chem_pot(nav_hf, beta, mu_old, method='nav_hf')
        print mu, mu_old, nav_new
        if abs(mu_new-mu_old) < de:
            break
        else:
            mu = mu_old
        it += 1

    return mu

