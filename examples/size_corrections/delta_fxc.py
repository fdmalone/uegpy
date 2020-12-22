#!/usr/bin/env python
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             '../../uegpy')))
import ueg as ue
import numpy as np
import pandas as pd
import infinite as inf
import size_corrections as szc
import structure as st
import scipy as sc
import utils as ut
import finite as fp

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
system = ue.UEG3D(rs, ne, ecut, zeta)

beta = 1.0 / (theta*system.ef)

# Chemical potential.
mu = inf.chem_pot(rs, beta, system.ef, zeta)
# magnitude of kvectors for finite system.
qvals = [system.kfac*(np.dot(k, k))**0.5 for k in system.kval[1:]]
# Minimum kvector
kmin = qvals[0] / system.kf
# Summation over discrete kvectors
v_sum = szc.v_summation(rs, theta, beta*mu, system.zeta, system.kf, nmax,
                        qvals, system.L)
# Integral up to maximum kvector considered
v_int = szc.v_integral(rs, theta, beta*mu, system.zeta, system.kf, 0, qvals[-1]/system.kf,
                       nmax)

# S(k_max)
s_kmax = st.rpa_matsubara(qvals[-1]/system.kf, rs, theta, beta*mu, zeta, nmax)

# f_xc sum.
f_xc_sum = szc.f_xc_summation(system, theta, beta*mu, nmax, qvals)
# # f_xc integral.
f_xc_int = inf.rpa_xc_free_energy(rs, theta, zeta, nmax, qvals[-1]/system.kf)
# f_x sum.
f_x_sum = szc.f_x_summation(rs, beta, mu, zeta, qvals, system.L)
# f_x int.
f_x_int = szc.f_x_integral(rs, beta, mu, zeta, qvals[-1])
# f_xc sum.
f_c_sum = szc.f_c_summation(system, theta, beta*mu, nmax, qvals)
# f_xc integral.
f_c_int = inf.rpa_correlation_free_energy(rs, theta, zeta, nmax, qvals[-1]/system.kf)

# f_c sum.


# f_c integral.

# Madelung constant.
mad = ut.madelung_approx(rs, ne)

names = ['Theta', 'rs', 'N', 'zeta', 'nmax', 'kf', 'kmin', 'kmax',
         'v_sum', 'v_int', 'f_xc_sum', 'f_xc_int', 'f_x_sum', 'f_x_int',
         'f_c_sum', 'f_c_int', 'mad', 'delta_v', 'delta_f_xc', 'delta_f_xc2',
         'BCDC', 'S_kmax']

frame = pd.DataFrame(data={'Theta': theta, 'rs': rs, 'N': ne, 'zeta': zeta,
                           'kf' : system.kf,
                           'nmax': nmax, 'kmin': [kmin/system.kf],
                           'kmax': [qvals[-1]/system.kf],
                           'v_sum': [v_sum],
                           'v_int': [v_int], 'mad': [0.5*mad],
                           'f_xc_sum': [f_xc_sum],
                           'f_xc_int': [f_xc_int],
                           'f_x_sum': [f_x_sum],
                           'f_x_int': [f_x_int],
                           'f_c_sum': [f_c_sum],
                           'f_c_int': [f_c_int],
                           'delta_v': v_int-(v_sum+0.5*mad),
                           'delta_f_xc': f_xc_int-(f_xc_sum+0.5*mad),
                           'delta_f_xc2': (f_x_int+f_c_int)-(f_x_sum+f_c_sum+0.5*mad),
                           'BCDC': szc.bcdc(rs, theta, zeta, ne),
                           'S_kmax': s_kmax}, columns=names)

print ("# Running uegpy version: %s"%(ut.get_git_revision_hash()))
import time
start = time.time()
print (frame.to_string(index=False))
print("# Time taken: {:f}".format(time.time()-start))
