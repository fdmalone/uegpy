#!/usr/bin/env python
'''Evaluate rpa correlation energy and exchange energy for given rs and theta.'''

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             '../../uegpy')))
import infinite as inf
import numpy as np
import utils as ut
import pandas as pd

theta = float(sys.argv[1])
rs = float(sys.argv[2])
zeta = 0
lmax = 200

beta = 1.0 / (ut.ef(rs, zeta)*theta)
mu = inf.chem_pot(rs, beta, ut.ef(rs, zeta), zeta)
f_c = inf.rpa_correlation_free_energy_dl(rs, theta, zeta, lmax)
f_x = inf.hfx_integral(rs, beta, mu, zeta)

frame = pd.DataFrame({'rs': rs, 'theta': theta, 'zeta': float(zeta), 'f_x': f_x,
                      'f_c': f_c}, index=[0],
                      columns=['rs', 'theta', 'zeta', 'f_x', 'f_c'])

print (frame.to_string(index=False))
