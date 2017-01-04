#!/usr/bin/env python

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             '../../uegpy')))
import ueg_sys as ue
import finite as fp
import numpy as np
import pandas as pd
import dielectric as di
import infinite as inf
import matplotlib.pyplot as pl


rs = float(sys.argv[1])
theta = float(sys.argv[2])
nel = int(sys.argv[3])
ecut = float(sys.argv[4])
zeta = int(sys.argv[5])

nvals = [14]

for n in nvals:
    system = ue.System(rs, n, ecut, zeta)
    o = 0
    beta = 1.0 / (system.ef * theta)
    mu = fp.chem_pot_sum(system, system.deg_e, beta)
    qvals = np.array([system.kfac * np.dot(k, k)**0.5 for k in system.kval[1:]])
    chi0 = [di.lindhard_matsubara_finite(system, q, mu, beta, 10) for q
            in system.kval[1:]]
    for (q, c) in zip(qvals, chi0):
        print q/system.kf, c
    pl.plot(np.array(qvals)/system.kf, chi0,
            label=r'Real: $N=%s$'%n, linewidth=0, marker='o')

pl.legend(numpoints=1)
pl.show()
pl.cla()
