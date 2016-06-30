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
from mpl_toolkits.mplot3d import Axes3D

theta = 0.625
rs = 1
zeta = 0

beta = 1.0 / (ut.ef(rs, zeta)*theta)

ef = ut.ef(rs, zeta)
kf = 2 * (ef)**0.5
omega = np.linspace(0, 2*ef, 10)
mu = inf.chem_pot(rs, beta, ut.ef(rs, zeta), zeta)

k = 1.1 * kf
qvals = np.linspace(0.1, 200, 100)
uvals = np.linspace(-1, 1, 100)

fig = pl.figure()
# ax = fig.add_subplot(111, projection='3d')
a = []
# for q in qvals:
    # for u in uvals:
    # a.append(se.f_qu(q*kf, k, -1, omega[9], beta, mu))

    # ax.scatter([q]*len(qvals), uvals, a)

# pl.plot(qvals, a)

im_sigma = [se.im_sigma_rpa(k, o, beta, mu) for o in omega]

pl.errorbar(omega/ef, im_sigma, fmt='o')
pl.show()
