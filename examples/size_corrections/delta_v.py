#!/usr/bin/env python
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             '../../uegpy')))
import ueg_sys as ue
import numpy as np
import pandas as pd
import infinite as inf
import size_corrections as szc
import structure as st
import scipy as sc
import utils as ut

if len(sys.argv[1:]) < 6:
    print ("Usage: size_corrections.py ne rs theta zeta nmax ecut")
    sys.exit()

ne = int(sys.argv[1])
rs = float(sys.argv[2])
theta = float(sys.argv[3])
zeta = int(sys.argv[4])
nmax = int(sys.argv[5])
ecut = float(sys.argv[6])

# Base system.
system = ue.System(rs, ne, ecut, zeta)

beta = 1.0 / (theta*system.ef)

# Chemical potential.
mu = inf.chem_pot(rs, beta, system.ef, zeta)
# magnitude of kvectors for finite system.
qvals = [system.kfac*(np.dot(k, k))**0.5 for k in system.kval[1:]]
# Minimum kvector
kmin = qvals[0]
# Correction from ommission of k = 0 term
delta_zero = szc.v_integral(rs, theta, beta*mu, system.zeta, system.kf, 0, kmin,
                            nmax)
delta_mad0 = szc.mad_integral(theta, beta*mu, system.zeta, system.kf, 0, kmin,
                            nmax)
# Summation over discrete kvectors
v_sum = szc.v_summation(rs, theta, beta*mu, system.zeta, system.kf, nmax,
                        qvals, system.L)
# Integral over potential
mad_int = szc.mad_integral(theta, beta*mu, system.zeta, system.kf, 0, qvals[-1],
                            nmax)
# Sum over potential
mad_sum = szc.mad_summation(theta, beta*mu, system.zeta, system.kf, nmax,
                        qvals, system.L)
# Integral up to maximum kvector considered
v_int = szc.v_integral(rs, theta, beta*mu, system.zeta, system.kf, 0, qvals[-1],
                       nmax)
# S(k_max)-1
s_kmax = st.rpa_matsubara_dl(qvals[-1], rs, theta, beta*mu, zeta, nmax)

# Madelung constant.
mad = ut.madelung_approx(rs, ne)

# Error from omission of k=0 point from vq*S(k)
delta_k0 = delta_zero - delta_mad0
delta_km = delta_zero - 0.5*mad

# Error between just vq*S(k)
delta_sk = (v_int - mad_int)  - (v_sum - mad_sum)

names = ['theta', 'rs', 'N', 'zeta', 'nmax', 'kmin', 'kmax', 'delta_zero',
         'delta_mad_zero', 'v_sum', 'v_int', 'mad', 'delta_mad', 'delta_v_int',
         'delta_v', 'delta_k0', 'delta_sk', 'delta_k0m', 'BCDC', 'S_kmax']

frame = pd.DataFrame(data={'theta': theta, 'rs': rs, 'N': ne, 'zeta': zeta,
                            'nmax': nmax, 'kmin': [kmin/system.kf], 'kmax':
                           [qvals[-1]/system.kf], 'delta_zero': [delta_zero],
                           'delta_mad_zero': delta_mad0, 'v_sum': [v_sum],
                           'v_int': [v_int], 'mad': [0.5*mad], 'delta_mad':
                           mad_int-mad_sum, 'delta_v_int': v_int-delta_zero,
                           'delta_v': v_int-(v_sum+0.5*mad), 'delta_k0':
                           delta_k0, 'delta_sk': delta_sk, 'delta_k0m':
                           delta_km, 'BCDC': szc.bcdc(rs, theta, zeta, ne),
                           'S_kmax': s_kmax}, columns=names)

print ("# Running uegpy version: %s"%(ut.get_git_revision_hash()))
print frame.to_string(index=False)
