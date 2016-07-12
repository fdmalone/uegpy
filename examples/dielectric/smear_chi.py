#!/usr/bin/env python
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             '../../uegpy')))
import ueg_sys as ue
import matplotlib.pyplot as pl
import numpy as np
import utils as ut
import infinite as inf
import dielectric as di
import self_energy as se

# im = np.array([-ut.vq(q)*di.im_lind(o, q, beta, mu) for o in omega])
# im = np.array([-ut.vq(q)*di.im_lind_smeared(o, q, beta, mu, delta=0.01, qmax=10*kf) for o in omega])
# re_kr = [di.kramers_kronig(im, omega, o, i, do=omega[1]) for (i, o) in enumerate(omega) ]
# re_kk = [di.kramers_kronig_int(o, q, beta, mu, omax=4*ef) for o in omega]

# pl.plot(omega/ef, im_eps_inv[:,1], label='Im-num')
# pl.plot(omega/ef, re_eps_inv[:,1], label='Re-num')

rs = 1.0
theta = 0.0625
zeta = 0
ef = ut.ef(rs, zeta)
beta = 1.0 / (theta*ef)
mu = inf.chem_pot(rs, beta, ef, zeta)
kf = (2*ef)**0.5
omega = np.linspace(0, 4*ef, 400)
qvals = np.linspace(0, 10*kf, 200)
(re_eps_inv, im_eps_inv) = se.tabulate_dielectric(beta, mu, 4*ef, 10*kf, 400, 40, delta=0.01)
im_sigma = [se.im_g0w0_self_energy(o, (2*o)**0.5, beta, mu, im_eps_inv, 200, 40, 10*kf, omega) for o in omega]

pl.plot(omega-ef, np.abs(im_sigma), label='Im Sigma')
# pl.plot(omega/ef, im_sigma2, label='More')
pl.xlim([-4, 4])
pl.legend()
pl.show()
