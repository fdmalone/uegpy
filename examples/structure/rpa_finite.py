#!/usr/bin/env python

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             '../../uegpy')))
import ueg_sys as ue
import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
import finite
import structure as st
import infinite as inf
import utils as ut
import dielectric as di

rs = 1.0
zeta = 0
theta = 1.0

ne = 14
lmax = 1000

system = ue.System(rs, ne, 10, zeta)
beta = 1.0 / (system.ef*theta)
mu = finite.chem_pot_sum(system, system.deg_e, beta)
mats = [st.rpa_finite(q, system, beta, mu, lmax) for q in system.kval[1:]]
qs = np.array([system.kfac*np.dot(q, q)**0.5 for q in system.kval[1:]])/system.kf
pl.errorbar(qs, mats, label=r'Matsubara', fmt='D')

mu = inf.chem_pot(rs, beta, system.ef, zeta)
qvals = np.linspace(0.001, 2*system.kf, 100)
sq = [st.rpa_matsubara(q/system.kf, rs, theta, beta*mu, zeta, lmax) for q in qvals]
pl.plot(qvals/system.kf, sq, '--')

pl.legend()
pl.show()
