''' Various approximate fits to the UEG or OCP '''

import numpy as np
import utils as ut
import scipy as sc

def classical_ocp(system, beta):
    ''' Evaluate the classical excess energy using the parametrised fit given by
    J. P. Hansen PRA 8, 6 1973.


Parameters
----------
system : class
    System being studied.
beta : float
    Inverse Temperature.

Returns
-------
U_xc : float
    Excess internal energy for classical OCP.

'''

    a1 = -0.895929
    b1 = 4.666486
    a2 = 0.113406
    b2 = 13.67541
    a3 = -0.908728
    b3 = 1.890560
    a4 = -0.116147
    b4 = 1.027755

    T = 1.0 / beta

    gamma = 1.0/(system.rs*T)
    U_xc = (1.5 * gamma**1.5 * (a1/(b1+gamma)**0.5 + a2/(b2+gamma)
            + a3/(b3+gamma)**1.5 + a4/(b4+gamma)**2))

    return U_xc


def ksdt(rs, t, zeta):
    '''
    Fit to RPIMC data of Brown et al (Phys. Rev. Lett. 110, 146405 (2013)) from
    Karasiev, Sjostrom, Dufty and Trickey, PRL 112, 076403. Please cite these
    guys.

    The KSDT fit is given as:

    .. math::
        f_{\mathrm{xc}}^{\zeta}(r_s, t)
        = -\\frac{1}{r_s}\\frac{\omega_{\zeta}a(t) + b_{\zeta}(t)r_s^{1/2}
        + c_{\zeta}(t)r_s}{1+d_{\zeta}(t)r_s^{1/2} + e_{\zeta}(t)r_s}

Parameters
----------
rs : float
    Density desired.
zeta : int
    Spin polarisation.
t : float
    Reduced temperature (:math:`T/T_F`)

Returns
-------
f_xc : float
    Exchange-correlation free energy per-particle.
'''

    if zeta == 1:
        w = 2.0**(1.0/3.0)
    else:
        w = 1.0

    # Functions used in fit.
    def a(t):

        return (0.610887*np.tanh(1.0/t)*(0.75+3.04363*t**2.0-0.09227*t**3.0
                +1.7035*t**4.0)/(1+8.31051*t**2.0+5.1105*t**4.0))


    def b(t, zeta):

        if zeta == 0:
            b1 = 0.283997
            b2 = 48.932154
            b3 = 0.370919
            b4 = 61.095357
            b5 = 0.871837
        else:
            b1 = 0.329001
            b2 = 111.598308
            b3 = 0.537053
            b4 = 105.086663
            b5 = 1.590438

        return (np.tanh(1.0/np.sqrt(t))*(b1+b2*t**2.0+b3*t**4.0)/
                (1.0+b4*t**2.0+b5*t**4.0))


    def c(t, zeta):

        if zeta == 0:
            c1 = 0.870089
            c2 = 0.193007
            c3 = 2.414644
        else:
            c1 = 0.848930
            c2 = 0.167952
            c3 = 0.088820

        return ((c1+c2*np.exp(-c3/t))*e(t, zeta))


    def d(t, zeta):

        if zeta == 0:
            d1 = 0.579824
            d2 = 94.537454
            d3 = 97.839603
            d4 = 59.939999
            d5 = 24.388037
        else:
            d1 = 0.551330
            d2 = 180.213159
            d3 = 134.486231
            d4 = 103.861695
            d5 = 17.750710

        return (np.tanh(1.0/np.sqrt(t))*(d1+d2*t**2.0+d3*t**4.0)/
                (1+d4*t**2.0+d5*t**4.0))


    def e(t, zeta):
        if zeta == 0:
            e1 = 0.212036
            e2 = 16.731249
            e3 = 28.485792
            e4 = 34.028876
            e5 = 17.235515
        else:
            e1 = 0.153124
            e2 = 19.543945
            e3 = 43.400337
            e4 = 120.255145
            e5 = 15.662836

        return (np.tanh(1.0/t)*(e1+e2*t**2.0+e3*t**4.0)/
                (1.0+e4*t**2.0+e5*t**4.0))


    return ((-1.0/rs)*(w*a(t)+b(t, zeta)*rs**(1.0/2.0)+c(t, zeta)*rs) /
                      (1+d(t, zeta)*rs**(1.0/2.0)+e(t, zeta)*rs))


def vwn_rpa(rs, zeta):
    ''' Vosko Wilk Nusair fit to RPA correlation energy of UEG. Can. J. Phys.
    58, 1200 (1980).

    Currently unpolarised only.

Parameters
----------
rs : float
    Density Parameter.
zeta : int
    Spin polarisation.

Returns
-------
ec : float
    Correlation energy.

'''

    b = 13.0720
    c = 42.7198
    x0 = -0.409286
    A = 0.0621814

    x = rs**0.5
    Q = (4*c - b**2.0)**0.5

    def X(q):
        return q**2.0 + b*q + c

    return (
        0.5 * A * (np.log(x**2.0/X(x)) + (2*b/Q) * np.arctan(Q/(2.0*x+b)) -
        (b*x0/X(x0)) * (np.log(((x-x0)**2.0)/X(x)) + (2*(b+2*x0)/Q)
        * np.arctan(Q/(2*x+b))))
    )



def pdw(rs, t, zeta):
    ''' Perrot, Dharma-wardana (Phys. Rev. A 30, 2619 (1984).)fit to RPA
    correlation free energy of unpolarised UEG.

Parameters
----------
rs : float
    Density Parameter.
zeta : int
    Spin polarisation.

Returns
-------
f_c : float
    Correlation free energy energy.

'''


    def c1(rs):

        return 10.9 / (1.0 + 0.00472*rs)

    def c2(rs):

        return (
            (39.5422 - 52.2381*rs**0.25 + 8.48554*rs**0.75) /
            (1 + 17.0999*rs**0.25)
        )

    def c3(rs):

        return (
            3.88860 / (1 + 0.133620*rs**0.5)
        )

    def c4(rs):

        return 0.122285 + 0.254281*rs**0.5

    def fh(rs, t):

        return -0.425437 * ((t/rs)**0.5) * np.tanh(1.0/t)


    return (
        (vwn_rpa(rs, zeta)*((1.0  + c1(rs)*t + c2(rs)*t**0.25)*np.exp(-c3(rs)*t))+fh(rs, t)*np.exp(-c4(rs)/t))
    )


def ti_STLS_params(t, zeta):
    '''Fitting parameters for :math:`T>0` STLS properties.

    Taken from Tanaka & Ichimaru J. Phys. Soc. Jpn. 55, 2278 (1986).

Parameters
----------
t : float
    Degeneracy temperature.
zeta : int
    Spin polarisation.

Returns
-------
(a, b, c, d, e) : floats
    Fitting parameters.
'''

    def a(t, zeta):
        if zeta == 1:
            l = (2./(9.*sc.pi))**(1./3.)
        else:
            l = (4./(9.*sc.pi))**(1./3.)

        return (1./(sc.pi*l)) * (0.75 + 3.04363*t**2 - 0.092270*t**3 +
                1.70350*t**4) * np.tanh(1./t) / (1. + 8.31051*t**2 + 5.1105*t**4)

    def b(t):
      return (t**(1./2.)*(0.323119 + 0.005348*t**(1./2.) +
              3.490430*t**(3./2.))/(1. + 0.000836*t + 4.03040*t**(2.)))

    def c(t):
      return (t*(0.514517 + 0.436502*t + 0.711644*t**(2.))/(1. + 1.86096*t**(2.)
              + 0.538374*t**(3.)))

    def d(t):
      return (t**(1./2.)*(0.549860 + 0.565967*t**(1./2.) - 1.15890*t +
              1.35663*t**(3./2.))/(1. - 0.651931*t + t**(2.)))

    def e(t):
      return (t*(0.636274 + 0.487840*t + 1.61592*t**(2.))/(1. + 2.36797*t**(2.) +
              1.09010*t**(3.)))

    return (a(t, zeta), b(t), c(t), d(t), e(t))


def ti_fxc(rs, t, pol):
    '''Excess free energy (over kT) from STLS.

    Taken from Tanaka & Ichimaru J. Phys. Soc. Jpn. 55, 2278 (1986).

Parameters
----------
t : float
    Degeneracy temperature.
zeta : int
    Spin polarisation.

Returns
-------
(a, b, c, d, e) : floats
    Fitting parameters.
'''

    G = ut.gamma(rs, t, pol)

    (a, b, c, d, e) = ti_STLS_params(t, pol)

    return (-1.0*((c/e)*G + (2./e)*(b-(c*d/e))*(G**(0.5)) +
            (1./e)*((a-(c/e))-(d/e)*(b-(c*d/e)))*np.log(abs(e*G+d*(G**(0.5))+1.)) -
            (2./(e*np.sqrt(4.*e-d*d)))*(d*(a-(c/e))+(2.-(d*d/e))*(b-(c*d/e))) *
            (np.arctan((2.*e*np.sqrt(G)+d)/np.sqrt(4.*e-d*d))-np.arctan(d/np.sqrt(4.*e-d*d)))))


def ti_txc(rs, t, pol):
    '''Excess kinetic energy (over kT) from STLS.

    Taken from Tanaka & Ichimaru J. Phys. Soc. Jpn. 55, 2278 (1986).

Parameters
----------
t : float
    Degeneracy temperature.
zeta : int
    Spin polarisation.

Returns
-------
(a, b, c, d, e) : floats
    Fitting parameters.

'''
    h = 0.00001 # Step size for numerical derivative.
    G = ut.gamma(rs, t, pol)

    return -t * (ti_fxc(ut.rs_gamma(G, t+h, pol), t+h, pol)-ti_fxc(rs, t, pol))/h


def ti_v(rs, t, pol):
    '''Potential energy from STLS.

    Taken from Tanaka & Ichimaru J. Phys. Soc. Jpn. 55, 2278 (1986).

Parameters
----------
t : float
    Degeneracy temperature.
zeta : int
    Spin polarisation.

Returns
-------
(a, b, c, d, e) : floats
    Fitting parameters.
'''

    (a, b, c, d, e) = ti_STLS_params(t, pol)
    G = ut.gamma(rs, t, pol)
    kT = ut.ef(rs, pol) * t

    return -kT * G * (a + b*G**(1./2.) + c*G)/(1. + d*G**(1./2.) + e*G)


def ti_uxc(rs, t, pol):
    '''Excess internal energy from STLS
    Taken from Tanaka & Ichimaru J. Phys. Soc. Jpn. 55, 2278 (1986).

Parameters
----------
t : float
    Degeneracy temperature.
zeta : int
    Spin polarisation.

Returns
-------
(a, b, c, d, e) : floats
    Fitting parameters.
'''

    kT = ut.ef(rs, pol) * t

    return kT*ti_txc(rs, t, pol) + ti_v(rs, t, pol)


def PDWParams():
    a1,a2,b1,b2,c1,c2,v,r,p,q,s,u,w = {},{},{},{},{},{},{},{},{},{},{},{},{}
    a1[1] = 5.6304
    a1[2] = 5.2901
    a1[3] = 3.6854
    b1[1] = -2.2308
    b1[2] = -2.0512
    b1[3] = -1.5385
    c1[1] = 1.7624
    c1[2] = 1.6185
    c1[3] = 1.2629
    a2[1] = 2.6083
    a2[2] = -15.076
    a2[3] = 2.4071
    b2[1] = 1.2782
    b2[2] = 24.929
    b2[3] = 0.78293
    c2[1] = 0.16625
    c2[2] = 2.0261
    c2[3] = 0.095869
    v[1] = 1.5
    v[2] = 3.0
    v[3] = 3.0
    r[1] = 4.4467
    r[2] = 4.5581
    r[3] = 4.3909
    p[1] = 0.653676
    p[2] = -0.157510
    p[3] = 0.190535
    q[1] = 0.166896
    q[2] = -0.308756
    q[3] = 0.691258
    s[1] = -0.373864
    s[2] = -0.144853
    s[3] = -0.890943
    u[1] = 0.472245
    u[2] = 2.495400
    u[3] = 5.656750
    w[1] = 1.0
    w[2] = 2.236068
    w[3] = 3.162278

    return a1,a2,b1,b2,c1,c2,v,r,p,q,s,u,w


def pdwfxc00(rs, theta, zeta):
    ''' Parametrisation of f_xc from CHNC data for the UEG.

    Ref: Perrot, Dharma-wardana, 62, 16536 (2000).

    .. Warning::
        This does not seem to match with Table. IV from the reference. Something
        is completely off for unpolarized case.

Paremeters
----------
rs : float
    Wigner-Seitz Radius.
theta : float
    Degeneracy temperature.
zeta : int
    Spin polarisation

Returns
-------
f_xc : float
    Exchange-correlation free energy.
'''

    a1,a2,b1,b2,c1,c2,v,r,p,q,s,u,w = PDWParams()

    g = lambda k: np.exp(5.*(rs-r[k]))
    z = lambda k: rs*(a2[k]+b2[k]*rs)/(1.+c2[k]*rs*rs)
    y = lambda k: v[k]*np.log(rs) + (a1[k] + b1[k]*rs + c1[k]*rs*rs)/(1. + rs*rs/5.)
    A = lambda k: np.exp((y(k) + g(k)*z(k))/(1.+g(k)))
    n = 3./(4.*sc.pi*(rs**3.))
    u1 = sc.pi*n/2.
    u2 = (2./3.)*np.sqrt(sc.pi*n)
    T = ut.calcT(rs, theta, zeta)
    P1 = (A(2)*u1 + A(3)*u2)*T*T + A(2)*u2*(T**(5./2.))
    P2 = 1. + A(1)*T*T + A(3)*(T**(5./2.)) + A(2)*(T**3.)
    exc0 = (inf.ex0(rs, zeta)+pz81(rs, zeta))
    fxc = (exc0 - P1)/P2

    h = (0.644291 + 0.0639443*rs)/(1. + 0.249611*rs)
    lam = 1.089 + 0.70*theta*np.sqrt(rs)
    F = lambda k: (p[k]+q[k]*(theta**(1./3.)))/(1. + s[k]*(theta**(1./3.)) + u[k]*(theta**(2./3.)))
    iB = 0.
    for [i,j,k] in [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]:
        iB += (np.sqrt(rs)-w[i])*(np.sqrt(rs)-w[j])/F(k)
    B = 1./iB
    alpha = 2 - h*np.exp(-theta*lam)
    Phi = ((1+zeta)**(alpha) + (1-zeta)**(alpha) - 2.)/(2.**(alpha) - 2.)

    return fxc*(1. + (2.**(B) - 1.)*Phi)


def pz81(rs, zeta):
    ''' Perdew-Zunger parametrization of the ground state correlation energy of the UEG.

    Ref: PRB, 23, 5048 (1981).

Parameters
----------
rs : float
    Wigner-Seitz radius
zeta : int
    Spin polarisation.

Returns
-------
E_c : float
    Correlation energy of 3D UEG parametrised to CA QMC data.
'''

    if rs > 1:
        if zeta == 1:
          B1 = 1.3981
          B2 = 0.2611
          G = -0.0843
        else:
          B1 = 1.0529
          B2 = 0.3334
          G = -0.1423

        return 2.*(G/(1+B1*np.sqrt(rs)+B2*rs))
    else:
        if zeta == 1:
          A = 0.01555
          B = -0.0269
          C = 0.0007
          D = -0.0048
        else:
          A = 0.0311
          B = -0.048
          C = 0.0020
          D = -0.0116

    return A*np.log(rs) + B + C*rs*np.log(rs) + D*rs
