#!/usr/bin/env python

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             '../../uegpy')))
import infinite as inf
import numpy as np
import utils as ut
import matplotlib.pyplot as pl
import self_energy as se

theta = 0.625
rs = 1
zeta = 0

beta = 1.0 / (ut.ef(rs, zeta)*theta)

ef = ut.ef(rs, zeta)
kf = 2 * (ef)**0.5
omega = np.linspace(0, 2*ef, 10)
mu = inf.chem_pot(rs, beta, ut.ef(rs, zeta), zeta)

k = 1.1 * kf

for q in np.linspace(0.01, 10, 10):
    a = []
    uvals = np.linspace(-1, 1, 10)
    for u in uvals:
        a.append(se.angular_integrand(u, k, q, 0.5, beta, mu))

    pl.plot(uvals, a)
    pl.show()

#im_sigma = [se.im_sigma_rpa(k, o, beta, mu) for o in omega]

pl.errorbar(omega/ef, im_sigma, fmt='o')
pl.show()
