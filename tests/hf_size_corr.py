#!/usr/bin/env python

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             '../uegpy')))
import size_corrections as sz
import matplotlib.pyplot as pl
import numpy as np

alpha = [0.1, 0.01, 0.001]
nmax = [int(np.sqrt(8*np.log(10)/a)) for a in alpha]
conv = []
#nmax = np.linspace(0, 100, 10)
#for n in nmax:
    #conv.append(sz.conv_fac(int(n), 0.001)[0])

#pl.plot(nmax, conv)
#pl.axvline(np.sqrt(8*np.log(10)/alpha))
#pl.show()

for (a, n) in zip(alpha, nmax):
    print a, n
    conv.append(sz.conv_fac(n, a)[0])

pl.plot(alpha, conv)
pl.xscale('log')
pl.axhline(2.8372)
pl.show()
