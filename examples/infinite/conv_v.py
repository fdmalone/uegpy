#!/usr/bin/env python

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             '../../uegpy')))
import ueg_sys as ue
import matplotlib.pyplot as pl
import numpy as np
import infinite as inf
import dielectric as di
import utils as ut
import pandas as pd

rs = 0.01
theta = 8
zeta = 0
lvals = np.linspace(100, 50000, 5)
qmax = [1, 2, 3, 4.01, 5, 8.01]

for q in qmax[-2:]:
    tan = [inf.rpa_v_tanaka(rs, theta, zeta, l, q) for l in lvals.astype(int)]
    pl.plot(lvals, tan, label='qmax=%s'%q)


pl.legend()
pl.show()
