#!/usr/bin/env python

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             '../../uegpy')))
import ueg_sys as ue
import finite as fp
import numpy as np
import pandas as pd

rs = float(sys.argv[1])
nel = int(sys.argv[2])
zeta = int(sys.argv[3])
theta = float(sys.argv[4])
ecut = float(sys.argv[5])
lmax = int(sys.argv[6])

system = ue.System(rs, nel, ecut, zeta)
b = 1.0 / (theta*system.ef)
mu = fp.chem_pot_sum(system, system.deg_e, b)
f_c = fp.rpa_correlation_free_energy(system, mu, b, lmax)
f_xm = fp.exchange_energy_chi0(system, mu, b, lmax)
f_x = fp.hfx_sum(system, b, mu)
frame = pd.DataFrame({'M': [len(system.kval)], 'lmax': [lmax],
                      'f_c': [f_c], 'f_xm': [f_xm], 'f_x': [f_x],
                      'r_s': [rs], 'zeta': [zeta], 'Theta': [theta], 'nel': [nel]},
                      columns=['r_s', 'M', 'nel', 'Theta', 'lmax', 'zeta', 'f_x', 'f_xm', 'f_c'])

print (frame.to_string(index=False, justify='right'))