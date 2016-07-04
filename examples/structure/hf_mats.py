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

tvals = [8, 1, 0.125, 0.0625]
qvals = np.linspace(0.01, 4, 100)
rs = 1
zeta = 0
ef = ut.ef(rs, zeta)
kf = (2.0*ef)**0.5

for t in tvals:

    ef = ut.ef(rs, 1)
    kf = (2.0*ef)**0.5
    beta = 1.0 / (t*ef)
    mu = inf.chem_pot(rs, beta, ef, 1)
    sq = [st.hartree_fock(q*kf, rs, beta, mu, 1) for q in qvals]
    ef = ut.ef(rs, 1)
    kf = (2.0*ef)**0.5
    beta = 1.0 / (t*ef)
    mu = inf.chem_pot(rs, beta, ef, 1)
    sq2 = [st.hartree_fock_matsubara(q, t, rs, 1) for q in qvals]

    pl.plot(qvals, np.array(sq2)-np.array(sq), label=r'Alt: $\Theta=%s$'%t, linestyle='--', marker='o')

zeta = 0
t = 1.0
ef = ut.ef(rs, zeta)
kf = (2.0*ef)**0.5
beta = 1.0 / (t*ef)
mu = inf.chem_pot(rs, beta, ef, zeta)
print (inf.hartree_fock_V(rs, t, zeta), inf.hfx_integral(rs, 1.0/ef, mu, zeta))

pl.legend(loc='best')
pl.show()
