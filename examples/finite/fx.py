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
# theta = np.logspace(-1, 1, 10)
theta = [0.00625]
ec = [2, 4, 6, 8]


f_x = []
f_x2 = []
M = []
e = 10
for e in ec:
    system = ue.System(1.0, 14, e, 0)
    sys.stderr.write('%s\n'%e)
    b = 1.0 / (theta[0]*system.ef)
    mu = fp.chem_pot_sum(system, system.deg_e, b)
    M.append(len(system.kval))
    f_x.append(fp.exchange_energy_chi0(system, mu, b, 100))
    f_x2.append(fp.hfx_sum(system, b, mu))

# frame = pd.DataFrame({'M': M, 'Theta': theta, 'f_x': f_x}, columns=['M',
    # 'Theta', 'f_x'])

# print (frame.to_string(index=False, justify='right'))
pl.errorbar(ec, f_x, fmt='o', label='mats')
pl.errorbar(ec, f_x2, fmt='D', markerfacecolor='None')
pl.xscale('log')
pl.legend()

pl.show()
