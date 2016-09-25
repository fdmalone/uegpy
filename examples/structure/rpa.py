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
import pandas as pd

qvals = np.append(np.linspace(0.00001, 0.1, 1000), (np.linspace(0.1, 10, 1000)))

zeta = 0
rs = float(sys.argv[1])
theta = float(sys.argv[2])
plot = sys.argv[3]
lmax = 10000

ef = ut.ef(rs, zeta)
kf = (2.0*ef)**0.5
beta = 1.0 / (theta*ef)
mu = inf.chem_pot(rs, beta, ef, zeta)
eta = beta * mu
sq = [st.rpa_matsubara(q, rs, theta, eta, zeta, lmax) for q in qvals]

frame = pd.DataFrame({'q': qvals, 'S_q': sq}, columns=['q', 'S_q'])

print (frame.to_string(index=False, justify='left'))

if (plot == 'True'):
    pl.plot(qvals, np.array(sq), 'mo', label=r'$\Theta=%s$'%theta)
    pl.legend(loc='best', numpoints=1)
    pl.show()
