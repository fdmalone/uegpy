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


rs = 1.0
theta = 0.00625
zeta = 0

ef = ut.ef(rs, zeta)
beta = 1.0 / (theta*ef)
mu = inf.chem_pot(rs, beta, ef, zeta)
kf = (0.1*ef)**0.5

q = kf

omega = np.linspace(0, 4*ef, 400)
kvals = np.linspace(0, 4*kf, 100)


#im = np.array([-ut.vq(q)*di.im_lind(o, q, beta, mu) for o in omega])
im = np.array([-ut.vq(q)*di.im_lind_smeared(o, q, beta, mu, delta=0.01, qmax=10*kf) for o in omega])
re = [1.0-ut.vq(q)*di.re_lind(o, q, beta, mu) for o in omega]
re_kr = [di.kramers_kronig(im, omega, o, omega_max=4*ef, do=omega[1], delta=0.1*omega[1]) for o in omega]

pl.plot

im_eps = np.zeros((40, 100))
i = 0 # frequency index
#for o in omega:
    #im_eps[i, :] = np.array([-ut.vq(q)*di.im_lind_smeared(o, q, beta, mu, delta=0.01, qmax=10*kf)
                                                                #for q in qvals])
    #re_eps[i, :] = di.kramers_kronig(im_eps[i,:], o, omega_max=10*ef)
    #denom = re_eps[i, :]**2.0 + im_eps[i, :]**2.0 + delta_freq
    #im_eps_inv[i, :] = im_eps[i, :] / denom
    #re_eps_inv[i, :] = re_eps[i, :] / denom
    #i += 1




pl.plot(omega/ef, im, label='Im')
pl.plot(omega/ef, re, label='Re')
pl.plot(omega/ef, re_kr, label='Re-KK', linestyle='--', linewidth=3)
pl.legend()
pl.show()

#new = [di.integrand2(k, 2*ef, q, beta, mu, delta=0.01)[0] for k in kvals]
#new2 = [di.integrand2(k, 2*ef, q, beta, mu, delta=0.1)[1] for k in kvals]
#pl.plot(kvals/kf, new)
#pl.plot(kvals/kf, new2, label='step')

#pl.legend()
#pl.show()
