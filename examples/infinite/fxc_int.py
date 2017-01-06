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

theta = 1
zeta = 0
minr = int(sys.argv[1])
maxr = int(sys.argv[2])
nr = int(sys.argv[3])
rs = np.logspace(-minr, maxr, nr)

v = np.array([inf.rpa_v_tanaka(r, theta, zeta, 1000, 5) for r in rs])

print 1.0/rs[-1]**2.0 * np.trapz(rs*v, rs)
pl.plot(rs, rs*v, 'ro')

pl.legend()
pl.show()
