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
import size_corrections as szc
import scipy.optimize as sc

def quad(x, a, b):

    return a * x + b

system = ue.System(1, 33, 10, 1)
qvals = np.linspace(0.0,0.1,100)
av = []
bv = []
bvals = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
frame = pd.DataFrame()

for b in bvals:

    rs = 1.0
    zeta = 1

    system = ue.System(rs, 33, 10, zeta)
    beta = b / system.ef

    mu = inf.chem_pot(rs, beta, system.ef, zeta)
    ft = [szc.hf_structure_factor(q*system.kf, rs, beta, mu, zeta) for q in
          qvals]
    pl.plot(qvals, ft, label=r'$\Theta=%s$'%(1.0/b), linestyle='--')
    pl.legend(numpoints=1, loc='best')
    pl.ylabel(r'$S(q)$')
    pl.xlabel(r'$q/q_F$')

    [a, c]= sc.curve_fit(quad, (qvals)**2.0, ft)[0]
    av.append(a)
    bv.append(c)
    pl.plot(qvals, quad(qvals**2.0, a, c))
    pl.show()

frame['Beta'] = bvals
frame['a'] = np.array(av)
frame['b'] = bv

print (frame.to_string(index=False))
