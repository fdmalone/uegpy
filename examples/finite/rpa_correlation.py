#!/usr/bin/env python

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             '../../uegpy')))
import ueg_sys as ue
import finite as fp
import numpy as np
import pandas as pd
import matplotlib.pyplot as pl

system = ue.System(1.0, 7, 10, 1)
theta = 1.0

b = 1.0 / (theta*system.ef)
lmax = 20
emax = [2.0, 4.0, 8.0, 10, 12, 14]

f_c = []
for e in emax:
    system = ue.System(1.0, 7, e, 1)
    mu = fp.chem_pot_sum(system, system.deg_e, b)
    f_c.append(fp.rpa_correlation_free_energy(system, mu, b, lmax))

pl.errorbar(emax, f_c, fmt='o')

pl.legend()
pl.show()
