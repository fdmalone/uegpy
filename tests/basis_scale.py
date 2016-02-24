#!/usr/bin/env python

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             '../uegpy')))
import ueg_sys as ue
import finite as fp
import numpy as np
import pandas as pd
import matplotlib.pyplot as pl


system = ue.System([1.0, 33, 10, 2])
bvals = [2**x for x in range(-5,5)]

bf = np.array(bvals) / system.ef

b = bf[2]

for ne in range(33, 200, 30):
    m = []
    mu = []
    u_0 = []
    print ne
    for ec in range(100, 1000, 100):

        system = ue.System([1.0, ne, ec, 2])
        m.append(len(system.spval))
        mu.append(fp.chem_pot_sum(system, system.deg_e, b))

    pl.plot(1.0/np.array(m), mu, label=r'ne=%s'%ne)
pl.legend(numpoints=1)
pl.ylabel(r'$\mu$ Ha')
pl.xlabel(r'$M^{-1}$')
pl.savefig('mu_ne_scaling.pdf', fmt='pdf')
