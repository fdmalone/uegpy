#!/usr/bin/env python

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             '../../uegpy')))
import ueg_sys as ue
import finite as fp
import numpy as np
import pandas as pd

system = ue.System(1.0, 33, 200, 2)

bvals = [2**x for x in range(-5,5)]

bvals = sorted(list(set(bvals+list(1.0/np.array(bvals)))))

bf = np.array(bvals) / system.ef

mu = [fp.chem_pot_sum(system, system.deg_e, b) for b in bf]
u_0  = [fp.energy_sum(b, m, system.deg_e, system.pol) / system.ne for (b, m) in
        zip(bf, mu)]

print pd.DataFrame({'Beta':bvals, 'mu':mu, 'u_0':u_0})
