''' Various approximate fits to the UEG or OCP '''

import numpy as np

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


def ksdt(rs, zeta, t):
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

        return 10.9 / (1.0 + 0.000472*rs)

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

        return -0.425437*(t/rs)**0.5 * np.tanh(1.0/t)


    return (
        vwn_rpa(rs, zeta)*(1.0  + c1(rs)*t + c2(rs)*t**0.25)*np.exp(-c3(rs)/t) +
        fh(rs, t)*np.exp(-c4(rs)/t)
    )
