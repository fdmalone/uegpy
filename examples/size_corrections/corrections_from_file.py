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
    print ("Usage: size_corrections.py ne rs theta zeta qmax file")
    sys.exit()

ne = int(sys.argv[1])
rs = float(sys.argv[2])
theta = float(sys.argv[3])
zeta = int(sys.argv[4])
qmax = float(sys.argv[5])
s_q = pd.read_csv(sys.argv[6], sep=r'\s+', comment='#')

# Base system.
# Need to work out cutoff needed to match with maximum qvector integrating to in
# evaluation of size correction.
ecut = 1
while True:
    system = ue.System(rs, ne, ecut, zeta)
    k = system.kfac * np.dot(system.kval[-1], system.kval[-1])**0.5
    if k/system.kf > qmax:
        break
    else:
        ecut = ecut + 10


beta = 1.0 / (theta*system.ef)

# magnitude of kvectors for finite system.
qvals = [(system.kfac/system.kf)*(np.dot(k, k))**0.5 for k in system.kval[1:]]

# Minimum kvector
kmin = qvals[0]
V = system.L**3.0

# Summation over discrete kvectors
v_sum = sum((1.0/(2.0*V)) * ut.vq(system.kf*q)*(np.interp(q, s_q.q, s_q.S_q) - 1.0) for q in qvals)

s_q = s_q[s_q.q < qmax]

# Integral up to maximum kvector considered
integrand = np.array([(sq-1.0) for (q, sq) in zip(s_q.q, s_q.S_q)])
v_int = (1.0/sc.pi) * np.trapz(integrand, s_q.q)

# Madelung constant.
mad = ut.madelung_approx(rs, ne)

names = ['Theta', 'rs', 'N', 'zeta', 'kmin', 'kmax',
         'v_sum', 'v_int', 'mad', 'delta_v', 'BCDC']

frame = pd.DataFrame(data={'Theta': theta, 'rs': rs, 'N': ne, 'zeta': zeta,
                            'kmin': [kmin/system.kf], 'kmax':
                            [qvals[-1]], 'v_sum': [v_sum], 'v_int':
                            [v_int], 'mad': [0.5*mad], 'delta_v':
                            v_int-(v_sum+0.5*mad),
                            'BCDC': szc.bcdc(rs, theta, zeta, ne)},
                            columns=names)

print ("# Running uegpy version: %s"%(ut.get_git_revision_hash()))
print frame.to_string(index=False, justify='left')
