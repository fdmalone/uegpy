''' Dielectric / Density response functions. '''
import numpy as np
import scipy as sc
import utils as ut
from scipy import optimize
import infinite as inf
from numpy import arctan2


def re_lind0(omega, q, kf):
    '''Real part of Lindhard Dielectric function at :math:`T = 0`.

    Note this is :math:`\mathrm{Re}\left[chi_{0\sigma}(\q, \omega)\\right]`

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

    Note this is :math:`\mathrm{Im}\left[chi_{0\sigma}(\q, \omega)\\right]`


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


def im_chi_rpa0(omega, q, kf, zeta):
    '''Imaginary part of :math:`T=0` rpa density-density response function.

Parameters
----------
omega : float
    frequency
q : float
    (modulus) of wavevector considered.
kf : float
    Fermi wavevector.
zeta : int
    Spin polarisation.

Returns
-------
chi_rpa : float
    Imaginary part of RPA density-density response function.

'''

    re = re_rpa_dielectric0(omega, q, kf, zeta)
    im = im_rpa_dielectric0(omega, q, kf, zeta)
    num = -im / ut.vq(q)
    denom = re**2.0 + im**2.0

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


def re_chi_rpa(omega, q, beta, mu):
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
    num = re_lind(omega, q, beta, mu)
    denom = ((1.0-vq*re_lind(omega, q, beta, mu))**2.0 +
                                          (vq*im_lind(omega, q, beta, mu))**2.0)

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

def re_rpa_dielectric(omega, q, beta, mu):
    ''' Real part of RPA dielectric function.'''

    re = 1.0 - (2-zeta)*ut.vq(q)*re_lind(omega, q, beta, mu)

    return re


def im_rpa_dielectric(omega, q, beta, mu):
    ''' Imaginary part of RPA dielectric function.

'''

    im = -(2-zeta)*ut.vq(q)*im_lind(omega, q, beta, mu)

    return im


def re_rpa_dielectric0(omega, q, kf, zeta):
    ''' Real part of the :math:`T=0` RPA dielectric function.

Parameters
----------
omega : float
    frequency
q : float
    (modulus) of wavevector considered.
kf : float
    Fermi wavevector.
zeta : int
    Spin polarisation.

Returns
-------
re_eps : float
    Real part of rpa dielectric function.

'''

    re = 1.0 - (2-zeta)*ut.vq(q)*re_lind0(omega, q, kf)

    return re


def im_rpa_dielectric0(omega, q, kf, zeta):
    ''' Imaginary part of the :math:`T=0` RPA dielectric function.

Parameters
----------
omega : float
    frequencydef im_chi_rpa_dandrea(omega, q, kf, rs, ef, theta, eta):

q : float
    (modulus) of wavevector considered.
kf : float
    Fermi wavevector.


Returns
-------
im_eps : float
    Imaginary part of rpa dielectric function.

'''

    im = - (2-zeta)*ut.vq(q)*im_lind0(omega, q, kf)

    return im


def lindhard_matsubara(x, rs, theta, eta, zeta, l):
    '''Dimensionless Lindhard function function factor.

    Taken from Tanaka and Ichimaru J.  Phys. Soc. Jap, 55, 2278 (1986).

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


def im_chi_tanaka(x, rs, theta, eta, zeta, l):
    ''' Imaginary part of RPA dielectric function in dimensionless form.

    Taken from Tanaka and Ichimary, Phys. Soc. Jap, 55, 2278 (1986).

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

    phi = lindhard_matsubara(x, rs, theta, eta, zeta, l)

    pre = 2.0*ut.gamma(rs, theta, zeta)*theta / (sc.pi*ut.alpha(zeta)*x**2.0)

    return phi / (1.0 + pre*phi)

def integrand2(k, omega, q, beta, mu, delta):

    ek = 0.5*k*k
    om_pl = omega + ek
    om_mi = - omega + ek
    qk = q*k


    #if (k < 1e-10): print om_pl, om_mi, (om_mi+qk)/delta
    return (
        #k*ut.fermi_factor(0.5*k*k, mu, beta) * (atan((om_mi+qk)/delta) -
        1.0/(sc.pi)*(arctan2((om_mi+qk), delta) -
                                                arctan2((om_mi-qk), delta) -
                                                arctan2((om_pl+qk), delta) +
                                                arctan2((om_pl-qk), delta)), #)#
        ut.step(k, np.abs(om_mi/q)) - ut.step(k, np.abs(om_pl/q))
    )


def integrand(k, omega, q, beta, mu, delta):

    eq = 0.5*q*q
    om_pl = omega + eq
    om_mi = - omega + eq
    qk = q*k


    #if (k < 1e-10): print om_pl, om_mi, (om_mi+qk)/delta
    return (
        k*ut.fermi_factor(0.5*k*k, mu, beta) *
        1.0/(sc.pi)*(arctan2((om_mi+qk), delta) -
                                                arctan2((om_mi-qk), delta) -
                                                arctan2((om_pl+qk), delta) +
                                                arctan2((om_pl-qk), delta))#
                        #ut.step(k, np.abs(om_mi/q)) - ut.step(k, np.abs(om_pl/q))
    )

def im_lind_smeared(omega, q, beta, mu, delta=0.01, qmax=10):

    I = sc.integrate.quad(integrand, 0, qmax, args=(omega, q, beta, mu, delta))[0]

    return - 1.0/(4.0*sc.pi*q) * I


def kramers_kronig(im_eps, omega, o, idx, do=0.01):

    if idx == 0 or idx == 1:
        I1 = 0
        integrand = omega[idx+1:]*im_eps[idx+1:]/(omega[idx+1:]**2.0-o*o)
        I2 = sc.integrate.simps(integrand[idx+1:], dx=do)
    elif idx+1 == len(omega):
        integrand = omega[:idx]*im_eps[:idx]/(omega[:idx]**2.0-o*o)
        I1 = sc.integrate.simps(integrand, dx=do)
        I2 = 0
    else:
        integrand = omega[:idx]*im_eps[:idx]/(omega[:idx]**2.0-o*o)
        I1 = sc.integrate.simps(integrand, dx=do)
        integrand = omega[idx+1:]*im_eps[idx+1:]/(omega[idx+1:]**2.0-o*o)
        I2 = sc.integrate.simps(integrand, dx=do)

    return 2.0/(sc.pi) * (I1+I2)

def kramers_kronig_eta(im_eps, omega, o, do, eta=0.00001):

    #integrand = im_eps / (omega - o + complex(0,eta))
    integrand = (omega-o)*im_eps / ((omega - o)**2.0 + eta**2.0)

    return (1.0/sc.pi) * sc.integrate.simps(integrand, dx=do)


def kramers_kronig_int(omega, q, beta, mu, omax):

    delta = 0.018

    def integrand(o2, omega, q, beta, mu):

        return -ut.vq(q)*o2*im_lind(o2, q, beta, mu)/(o2*o2-omega*omega)

    # conv = im_lind(omega, q, beta, mu)
    I1 = sc.integrate.quad(integrand, 0, omega-delta, args=(omega, q, beta, mu))[0]
    I2 = sc.integrate.quad(integrand, omega+delta, omax, args=(omega, q, beta, mu))[0]
    # I2 = -ut.vq(q)*conv*np.log((omax-omega)/omega)

    return 1.0 + 2/(sc.pi) * (I1+I2)


def lindhard_cplx_k(omega, ekq, ek, beta, mu, eta=0.01):

    n_k = ut.fermi_factor(ek, mu, beta)
    delta =  ek - ekq
    om = omega + complex(0, eta)

    return (n_k * (1.0/(om+delta) + 1.0/(-om+delta)))


def lindhard_cplx_finite(omega, iq, system, beta, mu, eta):

    chi0 = sum([lindhard_cplx_k(omega,
            0.5*system.kfac**2.0*np.dot(system.kval[iq]+system.kval[ik],
            system.kval[ik]+system.kval[iq]), system.spval[ik], beta, mu, eta)
            for ik in range(0, len(system.spval))])

    return (2-system.zeta)/(system.L**3.0) * chi0

def integrand3(k, omega, q, beta, mu, eta):

    om = omega + complex(0, eta)
    eq = 0.5*q*q
    dpl = eq - k*q
    dmi = eq + k*q

    return k * ut.fermi_factor(0.5*k*k, mu, beta) * (np.log((om+dpl)/(om+dmi)) + np.log((-om+dpl)/(-om+dmi)))

def lindhard_cplx(omega, q, beta, mu, zeta, eta=0.01):

    def integrand(k, omega, q, beta, mu, eta):

        om = omega + complex(0, eta)
        eq = 0.5*q*q
        dpl = eq - k*q
        dmi = eq + k*q

        return k * ut.fermi_factor(0.5*k*k, mu, beta) * (np.log((om+dpl)/(om+dmi)) + np.log((-om+dpl)/(-om+dmi)))

    def re_int(k, omega, q, beta, mu, eta):

        return sc.real(integrand(k, omega, q, beta, mu, eta))

    def im_int(k, omega, q, beta, mu, eta):

        return sc.imag(integrand(k, omega, q, beta, mu, eta))

    IR = sc.integrate.quad(re_int, 0, 10, args=(omega, q, beta, mu, eta))[0] 
    II = sc.integrate.quad(im_int, 0, 10, args=(omega, q, beta, mu, eta))[0] 

    return (2-zeta) / (8*sc.pi**2.0*q) * (IR + 1j * II) 
