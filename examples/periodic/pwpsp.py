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

a = 7.5233
nel = 12
atoms = a*np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])

print pc.total_energy(a, nel, atoms, 'al.ksp')
