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
import pandas as pd
import time

rs = float(sys.argv[1])
theta = float(sys.argv[2])
zeta = 0
ef = ut.ef(rs, zeta)
beta = 1.0 / (theta*ef)
mu = inf.chem_pot(rs, beta, ef, zeta)
kf = (2*ef)**0.5

start = time.time()

nkpoints = int(sys.argv[3])
nomega = int(sys.argv[4])
kmin = float(sys.argv[5])

omega = np.linspace(-10*ef, 10*ef, nomega)
qvals = np.linspace(kmin, 10*kf, nkpoints)

variables = ({'rs': rs, 'theta': theta, 'zeta': zeta, 'nkpoints': nkpoints,
              'nomega': nomega})

(re_eps_inv, im_eps_inv) = se.tabulate_dielectric_cplx(beta, mu, 10*ef, 10*kf,
                                                       nomega, nkpoints, zeta,
                                                       kmin=kmin, eta=abs(omega[1]-omega[0]))

se.write_table(im_eps_inv, omega, qvals, 'im_eps.csv', variables,
               calc_type='Dielectric function')

se.write_table(re_eps_inv, omega, qvals, 're_eps.csv', variables,
               calc_type='Real part of Dielectric function')
print ("Dielectic completed.")

ksel = np.linspace(0, 2*kf, 40)
im_sigma = np.zeros((len(omega), len(ksel)))
re_sigma = np.zeros((len(omega), len(ksel)))
A = np.zeros((len(omega), len(ksel)))
nk = []

for (ik, k) in enumerate(ksel):
    print ik, k
    im_sigma[:, ik] = [se.im_g0w0_self_energy(o, k, beta, mu, im_eps_inv,
                       nomega, nkpoints, 10*kf, omega) for o in omega]
    re_sigma[:, ik] = [se.hartree_fock(k, beta, mu, 10*ef) +
                       di.kramers_kronig_eta(im_sigma[:,ik], omega, o,
                                             omega[1]-omega[0]) for o in omega]
    A[:, ik] = se.spectral_function(im_sigma[:,ik], re_sigma[:,ik], mu, k, omega)
    nk.append(se.momentum_distribution(A[:,ik], beta, mu, k, omega))

se.write_table(im_sigma, omega, ksel, 'im_sigma.csv', variables,
               calc_type='Imaginary part of G0W0 self energy.')
se.write_table(re_sigma, omega, ksel, 're_sigma.csv', variables,
               calc_type='Real part of G0W0 self energy.')
se.write_table(A, omega, ksel, 'spectral_function.csv', variables,
               calc_type='G0W0 Spectral function.')

frame = pd.DataFrame({'k': ksel/kf, 'n_k': nk})

end = time.time()
print ("Time Take: %f s"%(end-start))
print (frame.to_csv('n_k.csv', index=False))
