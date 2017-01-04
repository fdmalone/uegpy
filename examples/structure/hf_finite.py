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
theta = 0.0625

ne = 14
lmax = 10

system = ue.System(rs, ne, 4, zeta)
beta = 1.0 / (system.ef*theta)
mu = finite.chem_pot_sum(system, system.deg_e, beta)
s = ([1.0 - (2.0-system.zeta)/ne *
      sum(ut.fermi_factor(0.5*system.kfac**2.0*np.dot(k+q, k+q), mu, beta)*
          ut.fermi_factor(0.5*system.kfac**2.0*np.dot(k, k), mu, beta)
          for k in system.kval) for q in system.kval[1:]])
mats = [-system.L**3.0/(system.ne*beta) *
        sum(di.lindhard_matsubara_finite(system, q,
    mu, beta, l) for l in range(-lmax, lmax+1)) for q in system.kval[1:]]
qs = np.array([system.kfac*np.dot(q, q)**0.5 for q in system.kval[1:]])/system.kf
pl.errorbar(qs, s, label=r'$N=%s$'%ne, fmt='o')
pl.errorbar(qs, mats, label=r'Matsubara', fmt='D')
# for q, m in zip(qs, mats):
    # print q, m

mu = inf.chem_pot(rs, beta, system.ef, zeta)
qvals = np.linspace(0.001, 4*system.kf, 100)
print sum((system.ne/(2.0*system.L**3.0))*ut.vq_vec(system.kfac*k)*(m-1.0) for (k, m) in zip(system.kval[1:], mats))
print sum((system.ne/(2.0*system.L**3.0))*ut.vq_vec(system.kfac*k)*(m-1.0) for (k, m) in zip(system.kval[1:], s))

sq = [st.hartree_fock(q, rs, beta, mu, zeta) for q in qvals]
pl.plot(qvals/system.kf, sq, '--')

pl.legend()
pl.show()
