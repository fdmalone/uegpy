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
theta = 0.000625
zeta = 0
ef = ut.ef(rs, zeta)
beta = 1.0 / (theta*ef)
mu = inf.chem_pot(rs, beta, ef, zeta)
kf = (2*ef)**0.5

nkpoints = 2000
nomega = 400
omega = np.linspace(0, 4*ef, nomega)
qvals = np.linspace(0, 10*kf, nkpoints)
(re_eps_inv, im_eps_inv, re_eps1, im_eps1) = se.tabulate_dielectric(beta, mu, 4*ef, 10*kf, nomega, nkpoints, zeta, delta=0.001)
re_eps = np.array([di.re_rpa_dielectric0(ef, q, kf, 0) for q in qvals])
im_eps = np.array([di.im_rpa_dielectric0(ef, q, kf, 0) for q in qvals])

#se.im_g0w0_self_energy(ef, (2*ef)**0.5, beta, mu, im_eps_inv, 200, 40, 10*kf, omega)
im_sigma = [se.im_g0w0_self_energy(o, (2*o)**0.5, beta, mu, im_eps_inv, nomega, nkpoints, 10*kf, omega) for o in omega]

u_grid = np.linspace(-1, 1, 100)

#pl.plot(u_grid, se.angular_integral(qvals[10], im_eps_inv[:,10], ef, (2*ef)**0.5, beta, mu, u_grid, omega)[1], label='Fermi')
#pl.plot(u_grid, se.angular_integral(qvals[10], im_eps_inv[:,10], ef, (2*ef)**0.5, beta, mu, u_grid, omega)[2], label='Step')
#pl.legend()
#pl.show()

pl.plot(omega/ef, np.abs(im_sigma), label='Im Sigma')
pl.show()

pl.plot(qvals, re_eps1[10,:], qvals, re_eps, 'ro')
pl.plot(qvals, im_eps1[10,:], qvals, im_eps, 'ro')
pl.show()
pl.plot(qvals, im_eps_inv[10,:], 'ro')
pl.plot(qvals, -im_eps1[10,:]/(re_eps1[10,:]**2.0+im_eps1[10,:]**2.0), label='Direct')
pl.plot(qvals, -im_eps/(re_eps**2.0+im_eps**2.0), label='Norm')
# pl.plot(omega/ef, im_sigma2, label='More')
pl.xlim([-4, 4])
pl.legend()
pl.show()
