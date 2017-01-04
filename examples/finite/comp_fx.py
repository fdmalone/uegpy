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

# theta = np.logspace(-1, 1, 10)
theta = [0.0625]
# lmax = np.array([10, 50, 100, 400, 1000])
lmax = np.array([100])

f_x = []
f_x2 = []
f_x3 = []
M = []
e = 4
system = ue.System(1.0, 14, e, 0)
b = 1.0 / (theta[0]*system.ef)
mu = fp.chem_pot_sum(system, system.deg_e, b)
M.append(len(system.kval))
pl.axhline(fp.hfx_structure(system, mu, b), label='HF')
for l in lmax:
    sys.stderr.write('%s\n'%l)
    f_x3.append(fp.exchange_energy_chi0(system, mu, b, l))

# frame = pd.DataFrame({'M': M, 'Theta': theta, 'f_x': f_x}, columns=['M',
    # 'Theta', 'f_x'])

# print (frame.to_string(index=False, justify='right'))
print f_x3
pl.errorbar(1.0/lmax, f_x3, fmt='x', label='Mats')
# pl.xscale('log')
pl.legend()

pl.show()
