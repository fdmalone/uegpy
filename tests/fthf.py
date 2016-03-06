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

bvals = [2**x for x in range(-10,5)]

system = ue.System(1, 33, 10, 1)

k = np.arange(0.1, 5, 0.1)
ek = 0.5*k**2.0

for b in bvals[::2]:
    pl.plot(k/system.kf, ek+inf.sigmax_discrete(k, ek, b/system.ef, inf.chem_pot(1,
            b, system.ef, 1), 1, 5, dq=0.1), label=r'$\Theta=%s$'%(1.0/b))
#pl.plot(k/system.kf, inf.t0_spect(k, system.kf), label=r'$\Theta=%s$'%(0), linestyle='--')
pl.legend(numpoints=1, loc='best')
pl.ylabel(r'$\varepsilon(k)$ (Ha)')
pl.xlabel(r'$k/k_F$')
pl.show()
