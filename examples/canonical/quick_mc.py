#!/usr/bin/env python

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             '../../uegpy')))
import ueg_sys as ue
import finite as fp
import infinite as inf
import numpy as np
import pandas as pd
import monte_carlo as mc
import utils as ut

rs = float(sys.argv[1])
ne = float(sys.argv[2])

bvals = np.logspace(-1,1.5, 30)

frame = []
t = 0

for b in bvals:
    c = max(20, ut.kinetic_cutoff(ne, 1.0/b))
    system = ue.System(rs, ne, c, 1)
    (f, time) = mc.sample_canonical_energy(system, b/system.ef, 100)
    t += time
    frame.append(f)

frame = pd.concat(frame)
print ("# Running uegpy version: %s"%(ut.get_git_revision_hash()))
print ("# Time taken: %s s"%time)
print (frame.to_string(index=False))
