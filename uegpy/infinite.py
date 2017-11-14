'''Evaluate properties of electron gas in thermodynamic limit using grand
   canonical ensemble'''
import numpy as np
import math
import scipy as sc
from scipy import integrate
import random as rand
from scipy import optimize
import uegpy.utils as ut
import uegpy.dielectric as di
import uegpy.structure as st


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

    return (
        sc.optimize.fsolve(nav_diff, ef, args=(beta, rs, zeta, nav, 1))[0]
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
    '''Derivative of integrand of standard Fermi integral :math:`I(eta, nu)`
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
    '''Standard Fermi integral :math:`I(\\eta, \\nu)`, where:

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


def ideal_kinetic_energy(beta, mu, rs, zeta):
    '''Total energy of ideal gas (Kinetic only):

    .. math::
        U_0 = (2-\\zeta) \\frac{2\\sqrt{2}}{3\\pi}r_s^3\\beta^{-5/2} I(3/2, \\eta)

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


def gc_free_energy(beta, mu, rs):
    '''Free energy:

    .. math::

        U = (2-\\zeta) \\frac{8\\sqrt{2}}{9\\pi}r_s^3\\beta^{-5/2} I(5/2, \\eta)

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
    Ideal grand potential.

'''

    return (
        -(2/3.) * (2-zeta) * ((4*sc.pi*rs**3.0)/3.0) * np.sqrt(2.0)/sc.pi**2.0
                        * beta**(-5./2.) * fermi_integral(5./2., mu*beta)
    )


def hfx_integrand(eta, power):
    '''Integrand of first order exchange contribution to internal energy.

    .. math::

        I_{-1/2}(\eta_0)^2

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


def f_x(rs, beta, mu, zeta):
    '''First-order exchange contribution to free energy:

    Uses the convenient form for the exchange integral from:
    Horovitz and Thieberger, Physica 71, 99 (1974).

    .. math::
        f_\mathrm{x} = -(2-\zeta) \\frac{r_s^3}{3\pi^2\\beta^2}
                       \int_{-\infty}^{\eta_0} d\eta I_{-1/2}(\eta)

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

    hfx = sc.integrate.quad(hfx_integrand, -np.inf, beta*mu, args=(2))[0]

    return - (2-zeta) * rs**3.0/(3*sc.pi**2.0*beta**2.0) * hfx


def mu_x(rs, beta, mu, zeta):
    '''First order exchange correction to the chemical potential.

    Turns out to be:

    .. math::

        \mu_\mathrm{x} = -\\frac{1}{\\sqrt{2}\\pi}\\beta^{-1/2}I_{-1/2}(\\eta_0)

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

    return -1.0/((2.0**0.5)*sc.pi) * beta**(-1./2.) * fermi_integral(-0.5, beta*mu)


def exchange_energy(rs, beta, mu, zeta):
    '''Exchange contribution to internal energy

    Turns out to be:

    .. math::

        u_\mathrm{x} = \\frac{3}{2} \mu_\mathrm{x} - f_{\mathrm{x}}


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
u_x : float
    Exchange energy.

'''

    u_x = 1.5 * mu_x(rs, beta, mu, zeta) - f_x(rs, beta, mu, zeta)

    return u_x


def rpa_correlation_free_energy(rs, theta, zeta, lmax, qmax):
    '''RPA correlation free energy.

    Calculated as given in Tanaka and Ichimaru, Phys. Soc.  Jap, 55, 2278 (1986)
    as:

    .. math::

        \\frac{\Omega_{\mathrm{c}}}{N} =
        -\\frac{1}{2(2\pi)^3n\\beta}\sum_l\int d\\mathbf{q}
        \Big[\log[1-v_{\\mathbf{q}}\chi^0(\\mathbf{q},z_l)]+
        v_{\\mathbf{q}}\chi^0(\\mathbf{q}, z_l)\Big]

Parameters
----------
rs : float
    Wigner-Seitz radius.
theta : float
    Degeneracy temperature.
zeta : int
    Spin polarisation.
lmax : int
    Maximum Matsubara frequency to include.

Returns
-------
f_c : float
    Exchange correlation free energy.
'''

    ef = ut.ef(rs, zeta)
    beta = 1.0 / (ef*theta)
    mu = chem_pot(rs, beta, ef, zeta)
    eta = beta * mu
    gamma = ut.gamma(rs, theta, zeta)
    alpha = ut.alpha(zeta)

    def integrand(x, rs, theta, eta, zeta, lmax, gamma, alpha):

        factor = (2-zeta)*gamma*theta/(sc.pi*alpha*x*x)

        i = 0
        for l in range(-lmax, lmax+1):

            eps_0 = factor * di.lindhard_matsubara(x, rs, theta, eta, zeta, l)
            i += x*x*(np.log(1+eps_0) - eps_0)

        return i

    integral = sc.integrate.quad(integrand, 0.0, qmax, args=(rs, theta, eta,
                                zeta, lmax, gamma, alpha))[0]

    return  1.5/((2.0-zeta)*beta) * integral


def rpa_xc_free_energy(rs, theta, zeta, lmax, qmax=6.0):
    '''RPA XC free energy as given in Phys. Soc. Jap, 55, 2278 (1986).

Parameters
----------
rs : float
    Wigner-Seitz radius.
theta : float
    Degeneracy temperature.
zeta : int
    Spin polarisation.
lmax : int
    Maximum Matsubara frequency to include.

Returns
-------
U_xc : float
    Exchange-Correlation energy.

'''

    ef = ut.ef(rs, zeta)
    beta = 1.0 / (ef*theta)
    mu = chem_pot(rs, beta, ef, zeta)
    eta = beta * mu
    gamma = ut.gamma(rs, theta, zeta)
    alpha = ut.alpha(zeta)

    def integrand(x, rs, zeta, theta, eta, gamma, lamb, lmax):

        return (
            x*x*(sum([np.log(1.0+(2-zeta)*gamma*theta/(sc.pi*lamb*x**2.0)
                           * di.lindhard_matsubara(x, rs, theta, eta, zeta, l))
                           for l in range(-lmax, lmax+1)])
                           - 2*(2-zeta)*gamma/(3*sc.pi*lamb*x**2.))
        )

    return (
        1.5/((2.0-zeta)*beta) * sc.integrate.quad(integrand, 0, qmax,
         args=(rs, zeta, theta, eta, gamma, alpha, lmax))[0]
    )


def rpa_v_tanaka(rs, theta, zeta, nmax=10000, qmax=5, correction=True):
    '''Evaluate RPA electron-electron (potential) energy.

    Taken from Tanaka & Ichimaru JPSJ 55, 2278 (1986) and benchmarked against this.

Parameters
----------
rs : float
    Wigner-Seitz radius.
theta : float
    Degeneracy temperature.
zeta : int
    Spin polarisation.
lmax : int
    Maximum Matsubara frequency to include.
qmax : float
    Maximum qvector to integrate up to before using asymptotic form.

Returns
-------
V : float
    Potential energy.
'''

    ef = ut.ef(rs, zeta)
    beta = 1.0 / (ef*theta)
    mu = chem_pot(rs, beta, ef, zeta)
    eta = beta * mu
    kf = (2.0*ef)**0.5

    def integrand(x, rs, theta, eta, zeta, nmax, kf):

            return (
                1.5 * theta * sum([di.im_chi_tanaka(x, rs, theta, eta, zeta, l)
                                  for l in range(-nmax, nmax+1)]) - 1.0
            )

    I = sc.integrate.quad(integrand, 0, qmax, args=(rs, theta,
                                   eta, zeta, nmax, kf))[0]
    if correction:
        # Evaulated analytically using asymptotic form for S(k) from TI.
        I += -8.0 / (9.0*sc.pi*kf*qmax**3.0)

    return  (
        ut.gamma(rs, theta, zeta)*I / (sc.pi*ut.alpha(zeta)*beta)
    )
