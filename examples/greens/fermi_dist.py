#!/usr/bin/env python

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             '../../uegpy')))
import infinite as inf
import numpy as np
import utils as ut
import greens_functions as gf
import matplotlib.pyplot as pl

theta = 0.00625
rs = 1
zeta = 0
lmax = 10000

beta = 1.0 / (ut.ef(rs, zeta)*theta)

n_k = []
kf = (2*ut.ef(rs,zeta))**0.5
kvals = np.linspace(0, 2*kf, 100)
mu = inf.chem_pot(rs, beta, ut.ef(rs, zeta), zeta)

fermi = []
for k in kvals:
    n_k.append(0.5+(1.0/beta)*sum([gf.G0_mats(beta, (0.5*k*k-mu), l)[0] for l in range(-lmax,lmax+1)]))
    fermi.append(ut.fermi_factor(0.5*k*k, mu, beta))
pl.errorbar(kvals/kf, np.array(fermi), fmt='o', linestyle='-')
pl.errorbar(kvals/kf, np.array(n_k), fmt='x')
pl.show()
