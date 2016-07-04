''' Structure factors at various levels of theory. '''
import numpy as np
import scipy as sc
import utils as ut
from scipy import optimize
import dielectric as di
import infinite as inf


def hartree_fock(q, rs, beta, mu, zeta):
    '''Static structure factor at Hartree--Fock level:

    .. math::
        S(q) = 1 - \\frac{r_s^3}{3\pi}
               \int_0^{\infty} dk k^2 f_k \int_{-1}^{1} du
               \\frac{1}{e^{\\beta(\\frac{1}{2}(k^2+q^2+2kqu)}+1}

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

    def integrand(k, q, mu, beta):

        return (
            k**2.0 * ut.fermi_factor(0.5*k**2.0, mu, beta) *
            sc.integrate.quad(ut.fermi_angle, -1.0, 1.0, args=(k, q, mu, beta))[0]
        )

    return (
        (1.0 - (2-zeta)*rs**3.0/(3.0*sc.pi) *
               sc.integrate.quad(integrand, 0, np.inf, args=(q, mu, beta))[0])
    )


def hartree_fock_ground_state(q, kf):
    '''Analytic static structure factor at Hartree--Fock level in the ground state:

    .. math::
        S(q) =
        \\begin{cases}
            \\frac{1}{2} \Big(\\frac{3}{4} \\frac{q}{q_F} - \\frac{1}{16}
            \Big(\\frac{q}{q_F}\Big)^3\Big) & \\text{if} \\ q \le 2q_F \\\\
            \\frac{1}{2} & \\text{if} \\ q > 2q_F
        \\end{cases}

Parameters
----------
q : float
    kvalue to calculate structure factor at.
kf : float
    Fermi wavevector.

Returns
-------
S(q) : float
    Static structure factor.

'''

    if (q <= 2*kf):
        return (3.0/4.0 * q/kf - 1.0/16.0 * (q/kf)**3.0)
    else:
        return 1


def hartree_fock_ground_state_integral(q, rs, kf):
    '''Static structure factor at Hartree--Fock level in the ground state:

    .. math::
        S(q) = 1 - \\frac{r_s^3}{3\pi}
               \int_0^{\infty} dk k^2 \\theta(k_F-k) \int_{-1}^{1} du
               \\theta(k_F-(k^2+2kqu+q^2))

Parameters
----------
q : float
    kvalue to calculate structure factor at.
rs : float
    Wigner-Seitz radius.
kf : float
    Fermi wavevector.
Returns
-------
S(q) : float
    Static structure factor.
'''

    def integrand(k, q, kf):

        return (k**2.0 *
               ut.step(kf, k) *
               sc.integrate.quad(ut.step_angle, -1, 1, args=(k, q, kf))[0])

    return (1 - (rs**3.0/(3.0*sc.pi)) *
                    sc.integrate.quad(integrand, 0, 2*kf, args=(q, kf))[0])


def rpa(q, beta, mu, rs):
    '''Finite temperature RPA static structure factor evulated as:

    .. math::
        S(q) = -\\frac{1}{\pi} \int_{-\infty}^{\infty}
            \mathrm{Im}[\chi^{\mathrm{RPA}}(q, \omega)] \coth(\\beta\omega/2)

    .. warning::
        This uses a naive approach which directly evaluates
        :math:`\mathrm{Im}[\chi^{\mathrm{RPA}}]`. Better results can be found
        using rpa_matsubara. In particular this routine will likely miss the
        plasmon contribution to the structure factor which dominates for some
        :math:`q_c(r_s, \Theta)` .

Parameters
----------
q : float
    (modulus) of wavevector considered.
beta : float
    Inverse temperature.
mu : float
    Chemical potential.
rs : float
    Density parameter.

Returns
-------
s_q : float
   Static structure factor.
'''

    def integrand(omega, q, beta, mu):

        return di.im_chi_rpa(omega, q, beta, mu) * 1.0/(np.tanh(0.5*beta*omega))

    return (
        -(4/3.)*rs**3.0 * sc.integrate.quad(integrand, 0, np.inf,
                                                args=(q, beta, mu))[0]
    )


def rpa_ground_state(q, kf, rs):
    '''Zero temperature RPA static structure factor.

    .. math::
        S(q) = -\\frac{1}{\pi} \int_{-\infty}^{\infty} d \\omega
            \mathrm{Im}[\chi^{\mathrm{RPA}}(q, \omega)]

Parameters
----------
q : float
    (modulus) of wavevector considered.

mu : float
    Chemical potential.

Returns
-------
s_q : float
   Static structure factor.
'''

    return (
        -(4/3.)*rs**3.0 * sc.integrate.quad(di.im_chi_rpa0, 0, np.inf,
                                                args=(q, kf))[0]
    )


def rpa_matsubara(q, theta, eta, zeta, kf, nmax):
    '''RPA static structure factor evaluated using matsubara frequencies.

    .. math::
        S(q) = -\\frac{1}{\pi} \int_{-\infty}^{\infty} d \\omega
            \mathrm{Im}[\chi^{\mathrm{RPA}}(q, \omega)]

Parameters
----------
q : float
    (modulus) of wavevector considered.
theta : float
    Degeneracy temperature.
eta : float
    :math:`\beta\mu`
zeta : int
    Spin polarisation.
kf : float
    Fermi Wavevector.
nmax : int
    Maximum number of matsubara frequencies to include.

Returns
-------
s_q : float
   Static structure factor.
'''

    sum_chi = sum([di.chi_rpa_matsubara(q, theta, eta, zeta, kf, n) for n in
                   range(-nmax, nmax+1)])

    return (
        - (1.5*(zeta+1)*sc.pi**2.0*theta/kf) * sum_chi
    )


def rpa_matsubara_dl(q, rs, theta, eta, zeta, lmax):
    '''RPA static structure factor evaluated using matsubara frequencies.

    .. math::
        S(q) = -\\frac{1}{\pi} \int_{-\infty}^{\infty} d \\omega
            \mathrm{Im}[\chi^{\mathrm{RPA}}(q, \omega)]

Parameters
----------
q/qf : float
    (modulus) of wavevector considered.
rs : float
    Wigner-Seitz radius.
theta : float
    Degeneracy temperature.
eta : float
    :math:`\beta\mu`
zeta : int
    Spin polarisation.
nmax : int
    Maximum number of matsubara frequencies to include.

Returns
-------
s_q : float
   Static structure factor.
'''

    sum_chi = sum([di.im_chi_tanaka(q, rs, theta, eta, zeta, l) for l in
                   range(-lmax, lmax+1)])

    return 1.5 * (zeta+1) * theta * sum_chi


def q0_plasmon(q, rs):
    ''' Plasmon structure factor.

    .. math::

        S(q) = \\frac{q^2}{2\\omega_p}

Parameters
----------
q : float
    Wavevector
rs : float
    Density parameter.

Returns
-------
S(q) : float
    Plasmon structure factor.
'''

    return q**2.0 / (2.0*(3.0/rs**3.0)**0.5)


def bijl_feynman(q, kf):
    ''' Bijl-Feynman structure factor.

Parameters
----------
q : float
    Wavevector
kf : float
    Fermi wavevector.

Returns
-------
S(q) : float
    Bijl-Feynman structure factor.
'''

    # Zeros of real part of RPA dielectric function.
    omega_q = sc.optimize.fsolve(di.re_rpa_dielectric, 0.5*kf**2.0, args=(q,
                                 kf))[0]

    return q**2.0 / (2*omega_q)
