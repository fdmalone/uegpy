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

tvals = np.logspace(-1,1)
rs = [0.1, 0.6, 1.0, 2.0]

frames = []

for r in rs:
    f_xc = [r*ft.ksdt(r, 1.0, t) for t in tvals]
    frames.append(pd.DataFrame({'Theta': tvals, 'f_xc': f_xc, 'rs': r}))

system = ue.System(1, 33, 10, 1)
mu = [inf.chem_pot(1, 1.0/(t*system.ef), system.ef, system.zeta) for t in tvals]
vx = np.array([inf.hfx_integral(system.rs, 1.0/(t*system.ef), m, system.zeta)
              for (t, m) in zip(tvals, mu)])

full = pd.concat(frames)
full = full.groupby('rs')
for r, d in full:
    pl.plot(d['Theta'], d['f_xc'], label=r'KSDT: $r_s$=%s'%r, linestyle='-')

pl.plot(tvals, vx, label=r'$f_{\mathrm{x}}$',
            linestyle='-.')
pl.legend(numpoints=1, loc='best')
pl.ylabel(r'$f_{\mathrm{xc}}\cdot r_s$ (Ha)')
pl.xlabel(r'$\Theta$')
pl.xscale('log')
pl.show()
