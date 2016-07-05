''' Dielectric / Density response functions. '''
import numpy as np
import scipy as sc
import utils as ut
from scipy import optimize
import infinite as inf


def re_lind0(omega, q, kf):
    '''Real part of Lindhard Dielectric function at :math:`T = 0`.

Parameters
----------
omega : float
    Frequency.
q : float
    (modulus) of wavevector considered.
kf : float
    Fermi wavevector.

Returns
-------
re_chi_0 : float
    Real part of Lindhard dielectric function.

'''

    qb = q / kf
    nu_pl = omega/(q*kf) + q/(2*kf)
    nu_mi = omega/(q*kf) - q/(2*kf)

    return (
        -(kf/(2*sc.pi**2))*(0.5
            - (1-nu_mi**2.0)/(4*qb)*np.log(np.abs((nu_mi+1)/(nu_mi-1)))
            + (1-nu_pl**2.0)/(4*qb)*np.log(np.abs((nu_pl+1)/(nu_pl-1))))
    )


def im_lind0(omega, q, kf):
    '''Imaginary part of Lindhard Dielectric function at :math:`T = 0`.

Parameters
----------
q : float
    (modulus) of wavevector considered.
omega : float
    Frequency.
kf : float
    Fermi wavevector.

Returns
-------
im_chi_0 : float
   Real part of Lindhard dielectric function.

'''

    qb = q / kf
    nu_pl = omega/(q*kf) + q/(2*kf)
    nu_mi = omega/(q*kf) - q/(2*kf)

    return (
        -1*((0.5*kf**2.0)/(4*sc.pi*q))*(ut.step(1, nu_mi**2.0)*(1-nu_mi**2.0) -
                                        ut.step(1, nu_pl**2.0)*(1-nu_pl**2.0))
    )


def im_chi_rpa0(omega, q, kf):
    '''Imaginary part of T=0 rpa density-density response function.

Parameters
----------
omega : float
    frequency
q : float
    (modulus) of wavevector considered.
kf : float
    Fermi wavevector.

Returns
-------
chi_rpa : float
    Imaginary part of RPA density-density response function.

'''

    vq = 4.0*sc.pi / q**2.0
    num = im_lind0(omega, q, kf)
    re = 1.0 - vq*re_lind0(omega, q, kf)
    im = -vq*im_lind0(omega, q, kf)
    denom = ((1.0-vq*re_lind0(omega, q, kf))**2.0 +
                                          (vq*im_lind0(omega, q, kf))**2.0)

    return num / denom


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
    Fermi wavevector.

Returns
-------
chi_rpa : float
    Imaginary part of RPA density-density response function.

'''

    vq = 4.0*sc.pi / q**2.0
    num = im_lind(omega, q, beta, mu)
    denom = ((1.0-vq*re_lind(omega, q, beta, mu))**2.0 +
                                          (vq*im_lind(omega, q, beta, mu))**2.0)

    return num / denom


def dandrea_real(omega, q, kf, rs, ef, theta, eta):
    '''Real part of rpa density-density response function.

Parameters
----------
omega : float
    frequency
q : float
    (modulus) of wavevector considered.
kf : float
    Fermi wavevector.
rs : float
    Density parameter.
ef : float
    Fermi energy.
theta : float
    Degeneracy temperature.
eta : float
    :math:`\beta\mu`.

Returns
-------
chi_rpa : float
    Imaginary part of RPA density-density response function.

'''
    def integrand(y, theta, x, eta):

        return  y / (1.0+np.exp(y**2.0/theta-eta)) * np.log(np.abs((x-y)/(x+y)))

    Q = q / (2.0*kf)
    z = omega / (4.0*ef)
    alpha = (4.0/(9.0*sc.pi))**(1.0/3.0)

    phi_pl = sc.integrate.quad(integrand, 0, np.inf, args=(theta, z/Q+Q, eta))[0]
    phi_mi = sc.integrate.quad(integrand, 0, np.inf, args=(theta, z/Q-Q, eta))[0]

    return 0.5*q**2.0 * alpha * rs / (16.0*sc.pi**2.0*Q**3.0) * (phi_pl - phi_mi)


def dandrea_im(omega, q, kf, rs, ef, theta, eta):
    '''Imaginary part of rpa density-density response function in reduced units.

Parameters
----------
omega : float
    frequency
q : float
    (modulus) of wavevector considered.
kf : float
    Fermi wavevector.
rs : float
    Density parameter.
ef : float
    Fermi energy.
theta : float
    Degeneracy temperature.
eta : float
    :math:`\beta\mu`.

Returns
-------
chi_rpa : float
    Imaginary part of RPA density-density response function.

'''

    Q = q / (2.0*kf)
    z = omega / (4.0*ef)
    alpha = (4.0/(9.0*sc.pi))**(1.0/3.0)

    return (
        - (2*sc.pi)**(-1.0)*(q**2.0 * alpha * rs * theta)/(32.0*Q**3.0) *
        np.log((1+np.exp(eta-(1.0/theta)*(z/Q-Q)**2.0))/(1+np.exp(eta-(1.0/theta)*(z/Q+Q)**2.0)))
    )


def im_chi_rpa_dandrea(omega, q, kf, rs, ef, theta, eta):
    '''Imaginary part of rpa density-density response function.

Parameters
----------
omega : float
    frequency
q : float
    (modulus) of wavevector considered.
kf : float
    Fermi wavevector.
rs : float
    Density parameter.
ef : float
    Fermi energy.
theta : float
    Degeneracy temperature.
eta : float
    :math:`\beta\mu`.

Returns
-------
chi_rpa : float
    Imaginary part of RPA density-density response function.

'''

    vq = 4.0*sc.pi / q**2.0
    num = dandrea_im(omega, q, kf, rs, ef, theta, eta)
    denom = ((1.0-vq*dandrea_real(omega, q, kf, rs, ef, theta, eta))**2.0 +
                      (vq*dandrea_im(omega, q, kf, rs, ef, theta, eta))**2.0)

    return num / denom


def re_lind(omega, q, beta, mu):
    '''Real part of free-electron Lindhard density-density response function.

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
re_chi : float
    Imaginary part of thermal Lindard function.

'''

    def integrand(k, beta, mu, q, omega):

        k_pl = (0.5*q**2.0 + omega) / q
        k_mi = (0.5*q**2.0 - omega) / q

        return (
            k * ut.fermi_factor(0.5*k**2.0, mu, beta) *
            (np.log(np.abs((k_pl+k)/(k_pl-k))) +
             np.log(np.abs((k_mi+k)/(k_mi-k))))
        )

    return (
        -(1.0/((2*sc.pi)**2.0*q)) * sc.integrate.quad(integrand, 0,
                                          np.inf, args=(beta, mu, q, omega))[0]
    )


def im_lind(omega, q, beta, mu):
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
        -1.0/(4*sc.pi*beta*q) * np.log((1.0+np.exp(-beta*(e_mi-mu)))/
                                         (1.0+np.exp(-beta*(e_pl-mu))))
    )


def re_rpa_dielectric(omega, q, kf):
    ''' Real part of RPA dielectric function.

Parameters
----------
omega : float
    frequency
q : float
    (modulus) of wavevector considered.
kf : float
    Fermi wavevector.


Returns
-------
re_eps : float
    Real part of rpa dielectric function.

'''

    re = 1.0 - ut.vq(q)*re_lind0(omega, q, kf)

    return re


def chi_rpa_matsubara(q, theta, eta, zeta, kf, n):
    ''' RPA density response function evaluated for complex (matsubara) frequencies.

Parameters
----------
q : float
    (modulus) of wavevector considered.
theta : float
    Degeneracy temperature.
eta : float
    :math:`\beta\mu`.
kf : float
    Fermi wavevector.
zeta : int
    Polarisation.
n : int
    nth Matsubara frequency.

Returns
-------
chi_rpa_n : float
    RPA dielectric function evaluated at nth matsubara frequency.

'''

    chi_n = lindhard_0_matsubara(q, theta, eta, zeta, kf, n)

    return chi_n / (1.0-ut.vq(q)*chi_n)


def lindhard_0_mats_n0(q, theta, eta, zeta, kf):
    ''' Lindhard function evaluated at zeroth matsubara frequency.

Parameters
----------
q : float
    (modulus) of wavevector considered.
theta : float
    Degeneracy temperature.
eta : float
    :math:`\beta\mu`.
zeta : int
    Polarisation.
kf : float
    Fermi wavevector.

Returns
-------
chi_rpa_n : float
    Lindhard function evaluated at zeroth matsubara frequency.

'''

    def integrand(y, x, theta, eta):

        return (
            y * ((y**2.0-0.25*x**2.0)*np.log(np.abs((2.0*y+x)/(2.0*y-x)))+x*y)
            / (2*(np.cosh(y**2.0/theta-eta)+1))
        )

    x = q / kf

    return (
        -kf / ((zeta+1)*sc.pi**2.0*x*theta) *
        sc.integrate.quad(integrand, 0, np.inf, args=(x, theta, eta))[0]
    )


def lindhard_0_matsubara(q, theta, eta, zeta, kf, n):
    '''Real part of free-electron Lindhard density-density response function.

Parameters
----------
q : float
    (modulus) of wavevector considered.
beta : float
    Inverse temperature.
mu : float
    Chemical potential.

Returns
-------
re_chi : float
    Imaginary part of thermal Lindard function.

'''

    if n == 0:

        prefactor = kf / (sc.pi**2.0*(zeta+1)*theta)

        def integrand(y, x, n, theta, eta):

            return (
                y * ((y**2.0-0.25*x**2.0)*np.log(np.abs((2.0*y+x)/(2.0*y-x)))
                    + x*y) / (2*(np.cosh(y**2.0/theta-eta)+1))
            )
    else:

        prefactor = kf / (2.0*sc.pi**2.0*(zeta+1))

        def integrand(y, x, n, theta, eta):

            return (
                y / (np.exp(y**2.0/theta-eta)+1.0)
                * np.log(np.abs(((2.0*sc.pi*n*theta)**2.0+(x**2.0+2.0*x*y)**2.0) /
                         ((2.0*sc.pi*n*theta)**2.0+(x**2.0-2.0*x*y)**2.0)))
            )

    x = q / kf
    chi_n = sc.integrate.quad(integrand, 0, np.inf, args=(x, n, theta, eta))[0]

    return - prefactor * chi_n / x


def tanaka(x, rs, theta, eta, zeta, l):
    ''' Dimensionless RPA dielectric function factor from Tanaka and Ichimaru J.
    Phys. Soc. Jap, 55, 2278 (1986).

    For large l or x we use the asyptotic form of

    .. math::
        phi(x, l) = \\frac{4 x^2}{3(2*\pi l \Theta)^2 + x^4} +
                     \mathcal{O}(x^{-4}, l^{-4})

Parameters
----------
x : float
    Momentum considered i.e., k/k_F.
rs : float
    Wigner-Seitz radius.
theta : float
    Degeneracy temperature.
eta : float
    :math:\beta*\mu
zeta : int
    Spin polarisation.
l : int
    Matsubara frequency.

Returns
-------
chi(x, l) : float
    Lindhard function evaluated at frequency l (for imaginary frequencies).
'''

    if np.abs(l) > 100 or x > 100:

        denom = (2*sc.pi*l*theta)**2.0 + x**4.0

        return (4/3.)*x*x / denom

    else:
        if l == 0:

            prefactor = 1.0 / (theta*x)

            def integrand(y, x, theta, eta, l):

                return (
                    y * ((y**2.0-0.25*x**2.0)*np.log(np.abs((2.0*y+x)/(2.0*y-x)))
                        + x*y) / (2*(np.cosh(y**2.0/theta-eta)+1))
                )
        else:

            prefactor = 1.0 / (2*x)

            def integrand(y, x, theta, eta, l):

                return (
                    y / (np.exp(y**2.0/theta-eta)+1.0)
                    * np.log(np.abs(((2.0*sc.pi*l*theta)**2.0+(x**2.0+2.0*x*y)**2.0) /
                             ((2.0*sc.pi*l*theta)**2.0+(x**2.0-2.0*x*y)**2.0)))
                )


        chi_n = sc.integrate.quad(integrand, 0, np.inf, args=(x, theta, eta, l))[0]

        return prefactor * chi_n / (zeta + 1)


def tanaka_large_l(x, rs, theta, eta, zeta, l):
    ''' Various orders of asymptotic forms for the Lindhard function'''

    denom = (2*sc.pi*l*theta)**2.0 + x**4.0

    t1 = (4/3.)*x*x / denom

    t2 = 8*x**4*theta**(5/2.) * inf.fermi_integral(1.5, eta) / denom**2.0

    t3 = (8/3.)*theta**(5/2.) * inf.fermi_integral(1.5, eta) * x**4 * (3*(2*sc.pi*l*theta)**2.0 - x**4) / denom**3.0

    return (t1, t2, t3)


def im_chi_tanaka(x, rs, theta, eta, zeta, l):
    ''' Imaginary part of rpa dielectric function in dimensionless form from:
    Tanaka and Ichimary, Phys. Soc. Jap, 55, 2278 (1986).

Parameters
----------
x : float
    Momentum considered i.e., k/k_F.
rs : float
    Wigner-Seitz radius.
theta : float
    Degeneracy temperature.
eta : float
    :math: `\beta*\mu`
zeta : int
    Spin polarisation.
l : int
    Matsubara frequency.

Returns
-------
Im(chi) : float
    Imaginary part of dielectric function in the RPA.
'''

    phi = tanaka(x, rs, theta, eta, zeta, l)

    pre = 2.0*ut.gamma(rs, theta, zeta)*theta / (sc.pi*ut.alpha(zeta)*x**2.0)

    return phi / (1.0 + pre*phi)
