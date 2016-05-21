#!/usr/bin/env python

import sys
import os
sys.path.append('/home/fm813/projects/uegpy/uegpy/')
sys.path.append('/home/fm813/projects/scripts/formatting/')
import ueg_sys as ue
import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
import infinite as inf
import size_corrections as szc
import structure as st
import scipy as sc
import formats as fmt

qmax = 4
beta = [16, 2, 1, 0.125]
zeta = 1
rs = 4

nmax = 1000

system = ue.System(rs, 33, 10, zeta)
frame = []
for b in beta:
    beta = b / system.ef
    mu = inf.chem_pot(rs, beta, system.ef, zeta)
    conv = [st.rpa_matsubara(qmax*system.kf, 1.0/b, beta*mu, system.zeta, system.kf, n) for n in nmax]
    frame.append(pd.DataFrame({'S_max': conv, 'n': nmax, 'beta': b},
                 columns=['beta', 'n', 'S_max']))

full = pd.concat(frame)

print (full.to_string(index=False))
