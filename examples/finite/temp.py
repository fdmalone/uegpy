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
theta = np.logspace(-1, 1, 10)
# theta = [4]


f_c = []
M = []
e = 10
system = ue.System(1.0, 14, e, 0)
for t in theta:
    sys.stderr.write('%s\n'%t)
    b = 1.0 / (t*system.ef)
    mu = fp.chem_pot_sum(system, system.deg_e, b)
    M.append(len(system.kval))
    f_c.append(fp.rpa_correlation_free_energy(system, mu, b, 20))

frame = pd.DataFrame({'M': M, 'Theta': theta, 'f_c': f_c}, columns=['M', 'Theta', 'f_c'])

print (frame.to_string(index=False, justify='right'))
pl.errorbar(theta, f_c, fmt='o')
pl.xscale('log')

pl.show()
