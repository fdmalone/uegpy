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

system = ue.System([0.1, 1024, 20, 2])
bvals = [2**x for x in range(-5,5)]

for b in bvals:
    u_old = 0
    M_old = 0
    for ec in np.logspace(2,3):
        system = ue.System([1.0, 1024, ec, 2])
        M_new = len(system.spval)
        if M_old != M_new:
            u_new = (fp.energy_sum(b, fp.chem_pot_sum(system,
                       system.deg_e, b), system.deg_e, system.pol))
            print u_new-u_old
            if u_new-u_old < 1e-6:
                print b, M_new, ec
                break
            u_old = u_new
            M_old = len(system.spval)

