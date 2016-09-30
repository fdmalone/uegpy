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
import matplotlib.pyplot as pl


rs = float(sys.argv[1])
theta = float(sys.argv[2])
nel = int(sys.argv[3])
ecut = float(sys.argv[4])
zeta = int(sys.argv[5])

system = ue.System(rs, nel, ecut, zeta)
beta = 1.0 / (system.ef * theta)
mu = fp.chem_pot_sum(system, system.deg_e, beta)

omegas = [0.25, 1.0, 2.0, 4.0]
qvals = [system.kfac/system.kf * np.dot(k, k)**0.5 for k in system.kval[1:]]

for o in omegas:
    chi0 = [di.lindhard_cplx_finite(o*system.ef, iq, system, beta, mu, 0.01) for (iq, q)
            in enumerate(qvals)]
    # chi0 = [di.im_lind_smeared(o*system.ef, q, beta, mu, 0.01) for (iq, q) in enumerate(qvals)]
    chi03 = [di.re_lind(o*system.ef, q, beta, mu) for (iq, q) in enumerate(qvals)]
    chi02 = [di.lindhard_cplx(o*system.ef, q, beta, mu, 0.01).real for (iq, q) in enumerate(qvals)]
    pl.plot(qvals, [c.real for c in chi0], 'ro', label=r'Real: $\omega=%s$'%o,
            markerfacecolor='None')
    # pl.plot(qvals, [c.imag for c in chi0], 'bD', label=r'Imag: $\omega=%s$'%o)
    # pl.plot(qvals, chi02, 'mv', markersize=5, label='complex TDL')
    pl.plot(qvals, np.array(chi02), 'rD', markerfacecolor='None', label='REAL TDL')
    pl.plot(qvals, np.array(chi03), 'rs', markerfacecolor='None', label='REAL TDL - 2')
    pl.legend(numpoints=1)
    pl.show()
    pl.cla()

