#!/usr/bin/env python

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             '../../uegpy')))
import infinite as inf
import numpy as np
import utils as ut


theta = 1.0
rs = 3.39
zeta = 0
lmax = 200

beta = 1.0 / (ut.ef(rs, zeta)*theta)
mu = inf.chem_pot(rs, beta, ut.ef(rs, zeta), zeta)
print (inf.rpa_xc_energy_tanaka(rs, theta, zeta, lmax), inf.rpa_correlation_free_energy_mats(rs, theta, zeta, lmax), inf.hfx_integral(rs, beta, mu, zeta))
