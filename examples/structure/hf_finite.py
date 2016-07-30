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
import utils as ut

tvals = np.logspace(-1, 1, 4)
system = ue.System(1, 33, 10, 1)
qvals = np.linspace(0.0,2,100)
qvals = np.append(qvals, np.linspace(2, 4, 10))

rs = 1.0
zeta = 1

ne = [7, 19, 33, 81, 203]

for n in ne:
    system = ue.System(rs, n, 3, zeta)
    s = ([1.0 - (1.0/n) * sum(ut.fermi_factor(0.5*system.kfac**2.0*np.dot(k+q, k+q),
        system.ef, 32*system.ef)*ut.fermi_factor(0.5*system.kfac**2.0*np.dot(k,
            k), system.ef, 32*system.ef) for k in system.kval) for q in system.kval[1:]])
    print s

    pl.errorbar(np.array([system.kfac*np.dot(q, q)**0.5 for q in system.kval[1:]])/system.kf, s, label=r'$N=%s$'%n, fmt='o')


pl.plot(qvals, [st.hartree_fock_ground_state(q*system.kf, system.kf) for q in qvals])
pl.legend()
pl.show()
