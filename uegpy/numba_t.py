#!/usr/bin/env python
'''Finite size corrections at various levels of approximation'''

from math import sqrt, exp
import sys
import time

def conv_fac(nmax, alpha):

    g = 0
    old = 0

    for ni in range(-nmax,nmax+1):
        for nk in range(-nmax,nmax+1):
            for nj in range(-nmax,nmax+1):
                n = ni**2 + nj**2 + nk**2
                old = g
                if n != 0:
                    g += exp(-alpha*n) / sqrt(n)
                diff = g - old

    return (2.0*3.14159/alpha-g, g, diff)

if __name__== '__main__':

   b = time.time()
   print (conv_fac(int(sys.argv[1]), float(sys.argv[2])))
   e = time.time()
   print (e-b)
