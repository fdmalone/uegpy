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

rs = 1
theta = 0.11
zeta = 0
lvals = np.linspace(0, 50, 5)

tan = [inf.rpa_correlation_free_energy_dl(rs, theta, zeta, l) for l in lvals.astype(int)]

pl.axhline(-0.1662*0.61089)

pl.plot(lvals, tan)

pl.show()
