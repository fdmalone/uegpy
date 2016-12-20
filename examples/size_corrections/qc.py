#!/usr/bin/env python
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             '../../uegpy')))
import ueg_sys as ue
import numpy as np
import pandas as pd
import infinite as inf
import size_corrections as szc
import structure as st
import scipy
import utils as ut


theta = [0.0625, 1, 8]
rs = np.logspace(-1, 1, 10)

for t in theta:
    for r in rs:
        scipy.optimize.solve(di.re_eps(ut.omega_p, kf, 
