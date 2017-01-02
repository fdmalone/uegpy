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

system = ue.System(1.0, 14, 10, 0)
theta = 0.625

b = 1.0 / (theta*system.ef)
lmax = [10, 20, 50, 80]
emax = [2.0, 4.0, 8.0]

f_c = []
M = []
e = 4
for e in emax:
    sys.stderr.write('%s\n'%e)
    system = ue.System(1.0, 14, e, 0)
    mu = fp.chem_pot_sum(system, system.deg_e, b)
    M.append(len(system.kval))
    f_c.append(fp.rpa_correlation_free_energy(system, mu, b, 20))

frame = pd.DataFrame({'M': M, 'f_c': f_c}, columns=['M', 'f_c'])

print (frame.to_string(index=False, justify='right'))
pl.errorbar(1.0/np.array(M), f_c, fmt='o')

pl.show()
