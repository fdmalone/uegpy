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

# Simple cubic lattice
basis = pc.Basis(0.5*np.array([1, 1, -1]), 0.5*np.array([-1, 1, 1]),
                 0.5*np.array([1, -1, 1]))

gamma = [0, 0, 0]
H = [0, 0, 2*sc.pi]
N = [0, sc.pi, sc.pi]
P = [sc.pi, sc.pi, sc.pi]

kx = np.linspace(0, 2*sc.pi, 100)

print kx
print basis.b1, basis.b2, basis.b3

ke = [pc.kinetic_energy(np.array([0, 0, k])) for k in kx]
ke2 = [pc.kinetic_energy(np.array([0, 0, k])+(basis.b3-basis.b1-basis.b2)) for k in kx]
ke3 = [pc.kinetic_energy(np.array([0, 0, k])+(basis.b2-basis.b1)) for k in kx]
ke4 = [pc.kinetic_energy(np.array([0, 0, k])+(basis.b1-basis.b2-basis.b3)) for k in kx]
# ke4 = [pc.kinetic_energy(np.array([0, 0, k])+basis.b3) for k in kx]

# pl.plot(kx, ke)
pl.plot(kx, ke2)
pl.plot(kx, ke3)
pl.plot(kx, ke4)

pl.show()
