#!/usr/bin/env python

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             '../../uegpy')))
import ueg_sys as ue
import finite as fp
import utils as ut
import monte_carlo as mc

time = float(sys.argv[1])
beta = float(sys.argv[2])
ne = float(sys.argv[3])

cutoff = ut.kinetic_cutoff(ne, 1.0/beta)

system = ue.System(0.1, ne, cutoff, 2)
t_per_it = mc.sample_canonical_energy(system, beta/system.ef, 10)[1] / 10

time = 0.9 * time
iterations = min(int(time/t_per_it), 1e6)

(frame, time) = mc.sample_canonical_energy(system, beta/system.ef, iterations)

print ("# Running uegpy version: %s"%(ut.get_git_revision_hash()))

print ("# Time taken: %s s"%time)

print frame.to_string(index=False)
