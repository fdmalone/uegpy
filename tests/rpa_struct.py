#!/usr/bin/env python

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             '../uegpy')))
import ueg_sys as ue
import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
import infinite as inf
import size_corrections as szc

system = ue.System(1, 33, 10, 1)

rs = 2
zeta = 1

system = ue.System(rs, 33, 10, zeta)
bvals = [2**n for n in range(-3, 5)]
qvals = np.linspace(0,0.5,30)
print bvals[3:4:2]

for b in bvals[3:4:2]:
    beta = b / system.ef
    mu = inf.chem_pot(rs, beta, system.ef, zeta)
    print b
    #omega = np.linspace(0, 5, 100)
    for q in qvals:
        s = [szc.rpa_structure_factor(q*system.kf, b, mu) for q in qvals]
    #s2 = [szc.rpa_structure_factor0(q*system.kf, system.kf, system.rs) for q in qvals]
        #s = [szc.im_chi_rpa0(o*2*system.ef, q*system.kf, system.kf) for o in omega]
        #print szc.rpa_structure_factor0(q*system.kf, system.kf)
    #s2 = [szc.im_chi_rpa(o, q*system.kf, beta, mu) for o in omega]
    #s = [1.0/(np.tanh(0.5*beta*o))*szc.im_chi_rpa(o*system.ef, q*system.kf, b, mu) for o in omega]
    #i = [szc.im_lind(b, mu, q*system.kf, o*system.ef) for o in omega]
    #i2 = [szc.im_lind0(q*system.kf, o*system.ef, system.kf) for o in omega]
    #r = [szc.re_lind(b, mu, q*system.kf, o*system.ef, ) for o in omega]
    #pl.plot(omega, r, omega, i, omega, i2, linestyle='--', label=r'$\Theta = %s$'%str(1.0/b))
        #pl.plot(omega, s, label=r'Finite T')
    pl.plot(qvals, s, label=r'Zero T')

#pl.legend(loc='best')
pl.show()
