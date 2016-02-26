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
import utils as ut

system = ue.System([1.0, 33, 10, 2])
b = 0.125 / system.ef

for ne in ut.magic_numbers(system)[2:8]:
    print ne
    m = []
    mu = []
    u_0 = []
    for ec in np.logspace(1.5,3):

        system = ue.System([1.0, ne, ec, 2])
        m.append(len(system.spval)/ne)
        u_0.append(fp.energy_sum(b, fp.chem_pot_sum(system,
                   system.deg_e, b), system.deg_e, system.pol))

    pl.plot(1.0/np.array(m), np.array(u_0)/ne, label=r'$N=%s$'%ne)

pl.legend(numpoints=1, loc='best')
pl.ylabel(r'$u_0$ (Ha)')
pl.xlabel(r'$(M/N)^{-1}$')
pl.show()
