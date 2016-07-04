#!/usr/bin/env python

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             '../../uegpy')))
import ueg_sys as ue
import matplotlib.pyplot as pl
import structure as st
import numpy as np
import utils as ut
import infinite as inf

qvals = np.linspace(0.1, 4, 10)

zeta = 0
rs = 1
theta = 1.0

ef = ut.ef(rs, zeta)
kf = (2.0*ef)**0.5
beta = 1.0 / (theta*ef)
mu = inf.chem_pot(rs, beta, ef, zeta)
eta = beta * mu
sq1 = [st.rpa_matsubara_dl(q, rs, theta, eta, zeta, 1000) for q in qvals]
sq2 = [st.rpa_matsubara(q*kf, theta, eta, zeta, kf, 1000) for q in qvals]

pl.plot(qvals, np.array(sq2))
pl.plot(qvals, np.array(sq1), label=r'Old: $\Theta=%s$'%theta, linewidth=0, marker='o')

pl.legend(loc='best')
pl.show()
