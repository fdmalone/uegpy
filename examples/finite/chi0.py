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

nvals = [38, 54, 204]

for n in nvals:
    system = ue.System(rs, n, ecut, zeta)
    o = system.ef
    beta = 1.0 / (system.ef * theta)
    mu = fp.chem_pot_sum(system, system.deg_e, beta)
    qvals = [system.kfac * np.dot(k, k)**0.5 for k in system.kval[1:]]
    #print system.kfac
    chi0 = [di.lindhard_cplx_finite(o*system.ef, iq, system, beta, mu, 0.662) for (iq, q)
            in enumerate(qvals)]
    pl.plot(np.array(qvals)/system.kf, [c.imag/2.0 for c in chi0], label=r'Real: $N=%s$'%n, linewidth=1, marker='o')

# chi0 = [di.im_lind_smeared(o*system.ef, q, beta, mu, 0.01) for (iq, q) in enumerate(qvals)]
print system.kfac
qvals = np.linspace(0, 8*system.kf, 100)
mu = inf.chem_pot(rs, beta, system.ef, zeta)
chi02 = [di.lindhard_cplx(o*system.ef, q, beta, mu, 0.01).imag for (iq, q) in enumerate(qvals)]
pl.plot(np.array(qvals)/system.kf, np.array(chi02), '--', markerfacecolor='None', label='REAL TDL')
pl.legend(numpoints=1)
pl.show()
pl.cla()
