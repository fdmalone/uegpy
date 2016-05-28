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
import utils as ut


ne = int(sys.argv[1])
system = ue.System(1, ne, 20, 1)
bvals = [2**x for x in range(-3,3)]

m = []
e = []

for b in bvals:
    u_old = 0
    M_old = 0
    for ec in np.logspace(1.4,3):
        system = ue.System(1, ne, ec, 1)
        M_new = len(system.spval)
        if M_old != M_new:
            u_new = (fp.energy_sum(b/system.ef, fp.chem_pot_sum(system,
                       system.deg_e, b/system.ef), system.deg_e, system.pol))
            # print u_new-u_old, ec
            if abs(u_new-u_old)/u_new < 1e-6:
                break
            u_old = u_new
            M_old = len(system.spval)
    m.append(M_new)
    e.append(ec)

frame = pd.DataFrame({'Beta': bvals, 'M': m, 'ec': e})

print (frame.to_string())
