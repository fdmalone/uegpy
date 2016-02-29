#!/usr/bin/env python

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             '../uegpy')))
import ueg_sys as ue
import finite as fp
import numpy as np
import pandas as pd
import monte_carlo as mc
import utils as ut

rs = float(sys.argv[1])
ne = float(sys.argv[2])
ecut = float(sys.argv[3])

iterations = 10000
system = ue.System(0.1, 33, ecut, 1)

bvals = [2**x for x in range(-3,1)]

frames = [mc.sample_canonical_energy(system, b/system.ef, iterations)[0] for
          b in bvals]

curr = pd.concat(frames)

cutoffs = [ut.kinetic_cutoff(ne, 1.0/b) for b in bvals]

corr = []
for (c, b) in zip(cutoffs, bvals):
    system = ue.System(rs, ne, c, 1)
    corr.append(mc.sample_canonical_energy(system, b/system.ef, iterations)[0])

corrected = pd.concat(corr)

for o in ['u', 't', 'v']:
    corrected = (
            ut.add_frame(corrected, curr, op='-', val1=o, label='d'+o, err=True)
    )

print (corrected.to_string(index=False))
