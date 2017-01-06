#!/usr/bin/env python

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             '../../uegpy')))
import ueg_sys as ue
import numpy as np
import pandas as pd
import finite
import structure as st
import infinite as inf
import utils as ut
import dielectric as di

rs = float(sys.argv[1])
nel = int(sys.argv[2])
zeta = int(sys.argv[3])
theta = float(sys.argv[4])
ecut = float(sys.argv[5])
lmax = int(sys.argv[6])

system = ue.System(rs, nel, ecut, zeta)
beta = 1.0 / (system.ef*theta)
mu = finite.chem_pot_sum(system, system.deg_e, beta)
mats = [st.rpa_finite(q, system, beta, mu, lmax) for q in system.kval[1:]]
qs = np.array([system.kfac*np.dot(q, q)**0.5 for q in system.kval[1:]])/system.kf

frame = pd.DataFrame({'rs': rs, 'Theta': theta, 'nel': nel, 'zeta': float(zeta), 'q/qf': qs,
                      's_q': mats}, columns=['rs', 'Theta', 'nel', 'zeta', 'q/qf',
                      's_q'])
print ("# Running uegpy version: %s"%(ut.get_git_revision_hash()))
print (frame.to_string(index=False))
