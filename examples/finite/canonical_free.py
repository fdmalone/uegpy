#!/usr/bin/env python

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             '../../uegpy')))
import ueg_sys as ue
import finite as fp
import numpy as np
import pandas as pd

system = ue.System(1.0, 7, 1, 1)

bvals = [0.0625, 0.125, 0.2, 0.25, 0.33, 0.4,  0.5,  0.725,  0.75, 0.8, 0.9, 1.0 ,1.5, 2, 2.5, 4, 8,
        16, 32]
bf = np.array(bvals) / system.ef

mu = [fp.chem_pot_sum(system, system.deg_e, b) for b in bf]
f0 = [fp.canonical_partition_function(system, b)[2] for b in bf]
u0 = [fp.canonical_partition_function(system, b)[0] for b in bf]

print pd.DataFrame({'Beta':bvals, 'u0': u0, 'f_0':f0})
