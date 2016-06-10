#!/usr/bin/env python

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             '../../uegpy')))
import ueg_sys as ue
import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
import infinite as inf
import structure as st

tvals = np.logspace(-1, 1, 4)
system = ue.System(1, 33, 10, 1)
qvals = np.linspace(0.0,2,100)
qvals = np.append(qvals, np.linspace(2, 4, 10))

rs = 1.0
zeta = 1

system = ue.System(rs, 33, 10, zeta)
bvals = [2**(-n) for n in range(-3, 5)]

for b in bvals:
    beta = b / system.ef
    mu = inf.chem_pot(rs, beta, system.ef, zeta)
    ft = [st.hartree_fock(q*system.kf, rs, beta, mu, zeta) for q in
          qvals]
    pl.plot(qvals, ft, label=r'$\Theta=%s$'%(1.0/b), linestyle='--')

gs = [st.hartree_fock_ground_state(q*system.kf, system.kf) for q in qvals]
pl.plot(qvals, np.array(gs), label=r'$\Theta=0$', linestyle='--')
pl.legend(numpoints=1, loc='best')
pl.ylabel(r'$S(q)$ a.u.')
pl.xlabel(r'$q/q_F$')
pl.show()
