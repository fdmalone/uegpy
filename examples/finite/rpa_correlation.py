#!/usr/bin/env python

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             '../../uegpy')))
import ueg_sys as ue
import finite as fp
import numpy as np
import pandas as pd

system = ue.System(1.0, 7, 10, 0)
theta = 1.0

b = 1.0 / (theta*system.ef)
lmax = 10

mu = fp.chem_pot_sum(system, system.deg_e, b)
f_c  = fp.rpa_correlation_free_energy(system, mu, b, lmax)

print f_c
