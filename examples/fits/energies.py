#!/usr/bin/env python

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             '../../uegpy')))
import ueg_sys as ue
import fits as ft
import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
import infinite as inf
import utils as ut

tvals = np.logspace(-1, 1, 100)
rs = [0.1, 0.5, 1.0, 2.0, 4.0]

zeta = 0
frames = []

for r in rs:
    ksdt = [ft.ksdt(r, t, zeta) for t in tvals]
    ksdt_u_xc = [ft.ksdt_uxc(r, t, zeta) for t in tvals]
    ksdt_v = [ft.ksdt_v(r, t, zeta) for t in tvals]
    stls = [ft.ti_fxc(r, t, zeta)*t*ut.ef(r, zeta) for t in tvals]
    stls_t_xc = [ft.ti_txc(r, t, zeta)*t*ut.ef(r, zeta) for t in tvals]
    stls_v = [ft.ti_v(r, t, zeta) for t in tvals]
    stls_u_xc = [ft.ti_uxc(r, t, zeta) for t in tvals]
    pdw =  [ft.pdw(r, t, zeta) for t in tvals]
    mu = [inf.chem_pot(r, 1.0/(t*ut.ef(r, zeta)), ut.ef(r, zeta), zeta) for t in tvals]
    f_x = [inf.f_x(r, 1.0/(t*ut.ef(r, zeta)), m, zeta) for (t, m) in zip(tvals, mu)]
    mu_x = [inf.mu_x(r, 1.0/(t*ut.ef(r, zeta)), m, zeta) for (t, m) in zip(tvals, mu)]
    u_x = [inf.exchange_energy(r, 1.0/(t*ut.ef(r, zeta)), m, zeta) for (t, m) in zip(tvals, mu)]
    frames.append(pd.DataFrame({'Theta': tvals, 'rs': r, 'ksdt_f_xc': ksdt,
                                'ksdt_u_xc': ksdt_u_xc, 'ksdt_v': ksdt_v,
                                'stls_f_xc': stls, 'stls_t_xc': stls_t_xc,
                                'stls_v': stls_v,
                                'stls_u_xc': stls_u_xc,'rpa_f_xc': pdw,
                                'mu': mu, 'f_x': f_x, 'u_x': u_x, 'mu_x': mu_x}))

print (pd.concat(frames).to_string(index=False, float_format='%E',
                                   justify='left'))
