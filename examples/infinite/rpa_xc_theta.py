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

rs = float(sys.argv[1])
zeta = 0
lmax = int(sys.argv[2])

ef = ut.ef(rs, zeta)

tvals = np.logspace(-1, 1, 20)
bvals = [1.0/(ef*t) for t in tvals]
mvals = [inf.chem_pot(rs, b, ef, zeta) for b in bvals]

f_c = [inf.rpa_correlation_free_energy(rs, t, zeta, lmax) for t in tvals]
f_x = [inf.f_x(rs, b, m, zeta) for (b, m) in zip(bvals, mvals)]
f_c_pdw = [ft.pdw(rs, t, zeta) for t in tvals]
f_xc_ksdt = [ft.ksdt(rs, t, zeta) for t in tvals]
v_rpa = [inf.rpa_v_tanaka(rs, t, zeta, lmax) for t in tvals]

frame = pd.DataFrame({'Theta': tvals, 'f_x': f_x, 'f_c': f_c, 'pdw': f_c_pdw,
                    'f_xc_ksdt': f_xc_ksdt, 'v_rpa': v_rpa},
                     columns=['Theta', 'f_x', 'f_c', 'pdw', 'f_xc_ksdt', 'v_rpa'])

print ("# Running uegpy version: %s"%(ut.get_git_revision_hash()))
print frame.to_string(index=False, justify='right')
