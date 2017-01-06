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
import fits as ft

rs = 1
theta = float(sys.argv[1])
zeta = 0
lvals = np.linspace(0, 100, 2)

tan = [inf.rpa_correlation_free_energy(rs, theta, zeta, l)/(-0.6109) for l in lvals.astype(int)]

print tan
pl.plot(lvals, tan)
pl.axhline(ft.pdw(rs, theta, 0)/-0.6109)

pl.show()
