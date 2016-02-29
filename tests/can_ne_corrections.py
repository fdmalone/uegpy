#!/usr/bin/env python

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             '../uegpy')))
import ueg_sys as ue
import finite as fp
import infinite as inf
import numpy as np
import pandas as pd
import monte_carlo as mc
import utils as ut

rs = float(sys.argv[1])
ne = float(sys.argv[2])
b = float(sys.argv[3])

c = max(20, ut.kinetic_cutoff(ne, 1.0/b))

system = ue.System(rs, ne, c, 1)
(frame, time) = mc.sample_canonical_energy(system, b/system.ef, 1000)

mu = inf.chem_pot(system.rs, b/system.ef, system.ef, system.zeta)
t0 = inf.energy_integral(b/system.ef, mu, system.rs, system.zeta)
vx = inf.hfx_integral(system.rs, b/system.ef, mu, system.zeta)

f2 = pd.DataFrame({'Beta': [b], 'rs': [system.rs], 't_inf': [t0], 'ux_inf': [vx]})


for (o, v) in zip(['t', 'ux'], ['t', 'v']):
    corrected = (
        ut.add_frame(f2, frame, op='-', val1=o+'_inf', val2=v, label='dN'+o)
    )
corrected =pd.merge(corrected, frame, on=['Beta', 'rs'])

print ("# Running uegpy version: %s"%(ut.get_git_revision_hash()))
print ("# Time taken: %s s"%time)
print (corrected.to_string(index=False))
