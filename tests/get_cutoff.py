#!/usr/bin/env python

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             '../uegpy')))
import utils as ut

ne = int(sys.argv[1])
theta = 1.0 / float(sys.argv[2])

cutoff = ut.kinetic_cutoff(ne, theta)

ec = max(25, cutoff)

print (ec)
