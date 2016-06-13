#!/usr/bin/env python

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             '../../uegpy')))
import ueg_sys as ue
import matplotlib.pyplot as pl
import numpy as np
import infinite as inf
import dielectric as di
import utils as ut

rs = 1
theta = 0.0625
zeta = 0
beta = 1.0 / (ut.ef(rs, zeta)*theta)
mu = inf.chem_pot(rs, beta, ut.ef(rs, zeta), zeta)
eta = beta * mu

xvals = np.linspace(1, 200, 100)

l = 1
lind = [di.tanaka(x, rs, theta, eta, zeta, l) for x in xvals]
(asy1, asy2, asy3) = np.array(zip(*[di.tanaka_large_l(x, rs, theta, eta, zeta, l) for x in xvals]))

pl.plot(xvals, np.abs(lind-asy1), label='first')
pl.plot(xvals, np.abs(lind-(asy1+asy2)), label='2nd')
pl.plot(xvals, np.abs(lind-(asy1+asy3)), label='3rd')

pl.yscale('log')
pl.xscale('log')
pl.legend(loc='best')

pl.show()
