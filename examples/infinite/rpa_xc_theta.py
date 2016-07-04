#!/usr/bin/env python

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             '../../uegpy')))
import infinite as inf
import numpy as np
import utils as ut
import pandas as pd
import fits as ft

rs = 1
zeta = 0
lmax = 200

ef = ut.ef(rs, zeta)

tvals = [2**x for x in range(-5,5)]
bvals = [1.0/(ef*t) for t in tvals]
mvals = [inf.chem_pot(rs, b, ef, zeta) for b in bvals]

f_c = [inf.rpa_correlation_free_energy_dl(rs, t, zeta, lmax) for t in tvals]
f_x = [inf.hfx_integral(rs, b, m, zeta) for (b, m) in zip(bvals, mvals)]
f_c_pdw = [ft.pdw(rs, t, zeta) for t in tvals]

frame = pd.DataFrame({'Theta': tvals, 'f_x': f_x, 'f_c': f_c, 'pdw': f_c_pdw},
                     columns=['Theta', 'f_x', 'f_c', 'pdw'])

print (frame.to_string(index=False))
