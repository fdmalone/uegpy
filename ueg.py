#!/usr/bin/env python
# -*- coding: utf-8 -*-

import ueg_sys as ueg_sys
import sys
import numpy as np
import matplotlib.pyplot as pl
import pandas as pd
import calcs as calcs

def main(args):
    ''' Main program

Params
------
args: list
    rs: float
        seitz radius.
    ne: integer
        number of electrons.
    pw: float
        plane wave cutoff measured in units of (2*pi/L)^2.
    pol: integer
        spin polarisation = 2 for fully polarised, 1 for unpolarised.
    calc_type: string
        possible calculation types include:
            All: perform all calculations
            partition: calculate the canonical partition function
            classical: calculate the classical excess energy of the one component plasma.
            test_root: something
            com: calculate the centre-of-mass correction to the total energy.
Returns
-------
data: pandas data frame
    contains columns of data for all requested quantities.
'''
    rs = float(args[0])
    ne = float(args[1])
    pw = float(args[2])
    pol = float(args[3])
    calc_type = args[4]

    system = ueg_sys.System(args)
    system.print_system_variables()
    data = calcs.run_calcs(system, calc_type)
    print data.to_string(index=False)

if __name__ == '__main__':

    main(sys.argv[1:])
