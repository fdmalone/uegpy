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
import scipy as sc

system = ue.System(1, 33, 10, 1)
qvals = np.linspace(0.1, 4, 100)
#qvals = np.append(qvals, np.linspace(2, 4, 10))

rs = [0.1, 1, 2]
zeta = 1


for r in rs:
    system = ue.System(r, 66, 10, zeta)
    pl.axvline((2.0*sc.pi/system.L)/system.kf)
    #print (2*sc.pi/system.L)/system.kf
    s = [szc.rpa_structure_factor0(q*system.kf, system.kf, r) for q in qvals]
    pl.plot(qvals, s, label=r'$r_s=$%s'%str(r))
    pl.legend(loc='best')

pl.show()
