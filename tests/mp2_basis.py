#!/usr/bin/env python

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             '../uegpy')))
import ueg_sys as ue
import finite as fn
import numpy as np
import pandas as pd
import monte_carlo as mc
import utils as ut
import matplotlib.pyplot as pl

rs = float(sys.argv[1])
ne = float(sys.argv[2])

m = [len(ue.System(rs, ne, ec, 0).kval) for ec in range(6, 40, 4)]

ec = [fn.mp2(ue.System(rs, ne, ec, 0)) for ec in range(6, 40, 4)]

pl.errorbar(1.0/np.array(m)**(1.0), ec, fmt='s')
pl.show()
