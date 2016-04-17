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
import size_corrections as sz

system = ue.System(1, 33, 10, 1)

rs = 1
zeta = 0

system = ue.System(rs, 33, 10, zeta)
b = 1.0
q = 1.115
omega = np.linspace(0, 10, 100)

beta = b / system.ef
mu = inf.chem_pot(rs, beta, system.ef, zeta)
print beta, mu

#zero = [sz.im_lind0(o*system.ef, q*system.kf, system.kf) for o in omega]
#rpa = [sz.im_lind(o*system.ef, q*system.kf, beta, mu) for o in omega]
#rpa_r = [sz.re_lind(o*system.ef, q*system.kf, beta, mu) for o in omega]
#dand = [sz.dandrea_im(o*system.ef, q*system.kf, system.kf, system.rs, system.ef, 1.0/(system.ef*beta), beta*mu) for o in omega]
#dand_r = [sz.dandrea_real(o*system.ef, q*system.kf, system.kf, system.rs, system.ef, 1.0/(system.ef*beta), beta*mu) for o in omega]
zero = [sz.im_chi_rpa0(o*system.ef, q*system.kf, system.kf) for o in omega]
rpa = [sz.im_chi_rpa(o*system.ef, q*system.kf, beta, mu) for o in omega]
dand = [sz.im_chi_rpa_dandrea(o*system.ef, q*system.kf, system.kf, system.rs, system.ef, 1.0/(system.ef*beta), beta*mu) for o in omega]

print sz.rpa_structure_factor(q*system.kf, beta, mu, rs), sz.rpa_structure_factor0(q*system.kf, system.kf, rs)

pl.plot(omega, zero, label=r'$T=0$')
pl.plot(omega, rpa, label=r'$T>0$')
pl.plot(omega, np.array(dand), label=r'$T>0$ Dandrea', linestyle='--')
#print np.array(dand)/np.array(rpa), np.array(dand_r)/np.array(rpa_r)
pl.legend()
pl.show()
