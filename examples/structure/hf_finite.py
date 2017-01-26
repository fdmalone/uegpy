#!/usr/bin/env python

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             '../../uegpy')))
import ueg_sys as ue
import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
import infinite as inf
import finite
import structure as st
import utils as ut

rs = 1.0
zeta = 0

system = ue.System(rs, 14, 10, zeta)
print system.rs, system.ne, system.zeta
beta = 1.0/(0.0625*system.ef)
mu = finite.chem_pot_sum(system, system.deg_e, beta)

sq = [st.hartree_fock_finite(system.kfac*q, system, mu, beta) for q in system.kval[1:]]

pl.plot(system.kfac*np.array([np.dot(q, q)**0.5 for q in system.kval[1:]]), sq,
        'ro')

vsq = system.ne/(2.0*system.L**3.0) * sum(ut.vq_vec(system.kfac*q)*(s-1.0) for
        (q, s) in zip(system.kval[1:], sq))

print(finite.hfx_sum(system, beta, mu), vsq)
pl.show()
