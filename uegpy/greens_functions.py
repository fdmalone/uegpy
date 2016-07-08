'''Module for evaluating Green's function for the electron gas'''

import numpy as np
import scipy as sc

def G0_mats(beta, ek, l):
    ''' Matsubara Green's function

Parameters
----------
beta : float
    Inverse temperature
ek : float
    single particle energy
l : integer
    Index of Matsubara frequency

Returns
-------
g0 : float
    Matsubara Green's function.
'''

    omega_n = sc.pi*(2*l+1)/beta
    re = - ek / (omega_n**2 + ek**2)
    im = -omega_n / (omega_n**2 + ek**2)

    return (re, im)

def re_G0(omega, ek):
    ''' Real part of retarded free electron  Green's function

Parameters
----------
omega : float
    frequency
ek : float
    single particle energy

Returns
-------
reg0 : float
    Real part of retarded free Green's function.
'''

    return 1.0 / (omega - ek)