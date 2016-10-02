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
import self_energy as se

rs = 5

theta = 0.000625
zeta = 0
beta = 1.0 / (ut.ef(rs, zeta)*theta)
mu = inf.chem_pot(rs, beta, ut.ef(rs, zeta), zeta)
eta = beta * mu

ef = ut.ef(rs, zeta)
kf = (2.0*ut.ef(rs, zeta))**0.5
q = 0.1
qvals = np.linspace(0.01, 3*kf, 1000)

omegas = np.linspace(-10*ef, 10*ef, 100)
print ef

re_chi = [1.0-(2-zeta)*ut.vq(q)*di.re_lind(o, q, beta, mu) for o in omegas]
im_chi2 = [-ut.vq(q)*di.im_lind_smeared(o, q, beta, mu, zeta, delta=0.001) for o in omegas]
chi = [di.lindhard_cplx(o, q, beta, mu, zeta, eta=0.001) for o in omegas]
(im, re, re_inv, im_eps_inv) = se.tabulate_dielectric(beta, mu, 2*ef, q, 100,
        2, 0, delta=0.001)
eps = [1.0 - ut.vq(q)*c for c in chi]
im_eps_inv2 = [(1.0 / e).imag for e in eps]

print eps[:10]

pl.plot(omegas/ef, re_chi, label='compare-real')
pl.plot(omegas/ef, im_chi2, label='compare')
pl.plot(omegas/ef, im_eps_inv[:,1], label='first')
pl.plot(omegas/ef, re[:,1], label='first')
op = ut.plasma_freq(rs)
pl.plot(omegas/ef, im_eps_inv2, label=r'Im$[\varepsilon^{-1}]$')
pl.plot(omegas/ef, [e.imag for e in eps], label='imag')
pl.plot(omegas/ef, [e.real for e in eps], label='real')

pl.legend(loc='best')

pl.show()
