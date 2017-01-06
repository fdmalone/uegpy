#!/usr/bin/env python
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             '../../uegpy')))
import ueg_sys as ue
import numpy as np
import pandas as pd
import infinite as inf
import size_corrections as szc
import structure as st
import scipy
import utils as ut
import dielectric as di
import matplotlib.pyplot as pl


theta = [0.0625, 1, 8]
rs = np.logspace(-1, 1, 10)
zeta = 0

frames = []

def bracket_root(beta, mu, zeta, kf, ef, qmin, qmax, tol=1e-8):

    n = 0
    while n < 100:
        c = 0.5*(qmin + qmax)
        (sol, dic, ierr, msg) = scipy.optimize.fsolve(di.re_eps, x0=5*ef,
                                            xtol=tol, full_output=True,
                                            args=(c, beta, mu, zeta))
        # print qmin, qmax, kf, c, n, ierr, sol[0]
        if abs(qmin-qmax) < tol:
            # print n, c, sol[0]
            return c
            break
        if ierr == 1:
            qmin = c
            n += 1
        elif ierr != 1:
            qmax = c
            n += 1

for t in theta:
    qcs = []
    for r in rs:
        beta = 1.0 / ut.calcT(r, t, zeta)
        mu = inf.chem_pot(r, beta, ut.ef(r, zeta), zeta)
        kf = ut.kf(r, zeta)
        ef = ut.ef(r, zeta)
        qmin = 0.005*kf
        qmax = 2*kf
        # Find wavevector above which re(eps) = 0 has no real roots.
        qmax = bracket_root(beta, mu, zeta, kf, ef, qmin, qmax)
        tol = 1e-8
        n = 0
        while n < 100:
            c = 0.5*(qmin + qmax)
            (sol, dic, ierr, msg) = scipy.optimize.fsolve(di.re_eps, x0=5*ef,
                                                xtol=tol, full_output=True,
                                                args=(c, beta, mu, zeta))
            omega_c = sol[0]
            fc = di.im_eps(omega_c, c, beta, mu, zeta)
            if abs(qmin-qmax) / kf < tol:
                qcs.append(c/kf)
                break
            elif abs(fc) > tol:
                qmax = c
                n += 1
            elif abs(fc) < tol:
                qmin = c
                n += 1
        # eps = [di.re_eps(q*ut.kf(r, zeta), ut.plasma_freq(r), beta, mu, zeta)
                # for q in np.linspace(0.001, 1, 400)]
        # ieps = [di.im_eps(q*ut.kf(r, zeta), ut.plasma_freq(r), beta, mu, zeta)
                # for q in np.linspace(0.001, 1, 400)]

        # print mu
        # eps = [di.re_eps(0.2*ut.kf(r, zeta), o, beta, mu, zeta)
                # for o in np.linspace(0.001, 2*ut.plasma_freq(r), 400)]
        # ieps = [di.im_eps(0.2*ut.kf(r, zeta), o, beta, mu, zeta)
                # for o in np.linspace(0.001, 2*ut.plasma_freq(r), 400)]
        # pl.plot(np.linspace(0.001, 2*ut.plasma_freq(r), 400)/ut.ef(r, zeta), eps)
        # pl.plot(np.linspace(0.001, 2*ut.plasma_freq(r), 400)/ut.ef(r, zeta), ieps, label='im')
    # pl.legend()
    # pl.show()
    frames.append(pd.DataFrame({'r_s': rs, 'Theta': [t]*len(rs), 'q_c': qcs}))

print pd.concat(frames)
