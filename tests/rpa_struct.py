#!/usr/bin/env python

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             '../uegpy')))
import ueg_sys as ue
import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
import infinite as inf
import size_corrections as szc

system = ue.System(1, 33, 10, 1)
qvals = np.linspace(0.1,2,100)
qvals = np.append(qvals, np.linspace(2, 4, 10))

rs = 2.0
zeta = 1

system = ue.System(rs, 33, 10, zeta)
bvals = [2**(-n) for n in range(-3, 5)]
qvals = np.linspace(0,1,10)
print qvals

for b in bvals[4:5:2]:
    print (b)
    beta = b / system.ef
    mu = inf.chem_pot(rs, beta, system.ef, zeta)
    omega = np.linspace(0,1,10)
    s = [szc.rpa_structure_factor(q*system.kf, b, mu) for q in qvals]
    pl.plot(qvals, s, label=r'$\Theta = %s$'%str(1.0/b))

pl.show()
