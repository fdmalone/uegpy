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

bvals = [2**x for x in range(-3,5)]

cutoffs = [max(20, ut.kinetic_cutoff(ne, 1.0/b)) for b in bvals]

t0 = []
vx = []
M = []
for (c, b) in zip(cutoffs, bvals):
    system = ue.System(rs, ne, c, 1)
    M.append(len(system.spval))
    mu = fp.chem_pot_sum(system, system.deg_e, b/system.ef)
    t0.append(fp.energy_sum(b/system.ef, mu, system.deg_e, system.pol))
    vx.append(fp.hfx_sum(system, b/system.ef, mu))

frame = pd.DataFrame({'Beta': bvals, 'ne': system.ne, 'M': M, 't': t0,
                      'ux': vx})

frame = ut.add_mad(system, frame)

t0 = []
vx = []
for b in bvals:
    mu = inf.chem_pot(system.rs, b/system.ef, system.ef, system.zeta)
    t0.append(inf.energy_integral(b/system.ef, mu, system.rs, system.zeta))
    vx.append(inf.hfx_integral(system.rs, b/system.ef, mu, system.zeta))

f2 = pd.DataFrame({'Beta': bvals, 'rs': system.rs, 't_inf': t0, 'ux_inf': vx})

for o in ['t', 'ux']:
    corrected = (
        ut.add_frame(f2, frame, op='-', val1=o+'_inf', val2=o, label='dN'+o)
    )
corrected =pd.merge(corrected, frame, on='Beta')

print (corrected.to_string(index=False))
