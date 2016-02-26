#!/usr/bin/env python

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             '../uegpy')))
import ueg_sys as ue
import fits as ft
import matplotlib.pyplot as pl
import numpy as np
import pandas as pd

tvals = np.logspace(-2,2)
rs = [0.1, 0.6, 1.0, 2.0]

frames = []

for r in rs:
    f_xc = [r*ft.ksdt(r, 1.0, t) for t in tvals]
    frames.append(pd.DataFrame({'Theta': tvals, 'f_xc': f_xc, 'rs': r}))

full = pd.concat(frames)
full = full.groupby('rs')
for r, d in full:
    pl.plot(d['Theta'], d['f_xc'], label=r'$r_s$=%s'%r)
pl.legend(numpoints=1, loc='best')
pl.ylabel(r'$f_{\mathrm{xc}}\cdot r_s$ (Ha)')
pl.xlabel(r'$\Theta$')
pl.xscale('log')
pl.show()
