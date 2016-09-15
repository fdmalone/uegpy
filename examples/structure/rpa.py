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

qvals = np.linspace(0.1, 4.01, 40)

zeta = 0
rs = 1
theta = 0.0625

ef = ut.ef(rs, zeta)
kf = (2.0*ef)**0.5
beta = 1.0 / (theta*ef)
mu = inf.chem_pot(rs, beta, ef, zeta)
eta = beta * mu
sq = [st.rpa_matsubara(q, rs, theta, eta, zeta, 10000) for q in qvals]

frame = pd.DataFrame({'q': qvals, 'S_q': sq}, columns=['q', 'S_q'])

print (frame.to_string(index=False, justify='left'))

pl.plot(qvals, np.array(sq), label=r'$\Theta=%s$'%theta)

pl.legend(loc='best')
pl.show()
