#!/usr/bin/env python

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             '../../uegpy')))
import ueg_sys as ue
import utils as ut
import periodic as pc
import scipy as sc
import numpy as np
import matplotlib.pyplot as pl

# FCC Lattice
basis = pc.Basis(0.5*np.array([1, 1, 0]), 0.5*np.array([0, 1, 1]),
                 0.5*np.array([1, 0, 1]))

kz = np.linspace(0, 2*sc.pi, 100)

nmax = 3
for m1 in range(-nmax, nmax):
    for m2 in range(-nmax, nmax):
        for m3 in range(-nmax, nmax):
            gn = m1*basis.b1 + m2*basis.b2 + m3*basis.b3
            ek = ([0.5*np.dot(np.array([0, 0, k])+gn, np.array([0, 0, k])+gn) for
                  k in kz])
            pl.plot(kz, ek)

pl.show()
