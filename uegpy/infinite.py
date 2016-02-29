'''Evaluate properties of electron gas in thermodynamic limit using grand
   canonical ensemble'''
import numpy as np
import math
import scipy as sc
from scipy import integrate
import random as rand
from scipy import optimize


def chem_pot(rs, beta, ef, zeta):
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

    # System density.
    rho = ((4*sc.pi*rs**3.0)/3.0)**(-1.0)

    return (
        sc.optimize.fsolve(nav_diff, ef, args=(beta, rho, zeta))[0]
    )


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


def nav_diff(mu, beta, rho, zeta):
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
    return (nav(beta, mu, zeta) - rho)


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


def hfx_integrand(eta):
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

    return fermi_integral(-0.5, eta)**2


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
