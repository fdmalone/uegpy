''' System properties.
'''

import scipy as sc
import numpy as np
import math
import json

class System:
    '''System Properties - Everything is measured in Hartree atomic units.

Attributes
----------
rs : float
    Wigner-Seitz radius.
ne : int
    Number of electrons.
ecut : float
    Plane wave cutoff.
zeta : int
    Spin polarisation = 1 for fully polarised case, 0 for unpolarised.
L : float
    Box length.
kfac : float
    kspace grid spacing.
kf : float
    Fermi wavevector.
ef : float
    Fermi energy
spval : list
    containing single particle eigenvalues, sorted in increasing value of
    kinetic energy.
kval : list
    Plane wave basis vectors, sorted in increasing value of kinetic energy.
deg_e : list of lists
    Compressed list of spval containing unique eigenvalues and their
    degeneracy.
M : int
    Number of plane waves in our basis.
'''

    def __init__(self, rs, ne, ecut, zeta):
        '''Initialise system.

        Parameters
        ----------
        rs : float
            Wigner-Seitz radius.
        ne : int
            Number of electrons.
        ecut : float
            Plane wave cutoff.
        zeta : int
            Spin polarisation = 1 for fully polarised case, 0 for unpolarised.
        '''

        # Seitz radius.
        self.rs = rs
        # Number of electrons.
        self.ne = ne
        # Kinetic energy cut-off.
        self.ecut = ecut
        # Spin polarisation.
        self.zeta = zeta
        # Box Length.
        self.L = self.rs*(4*self.ne*sc.pi/3.)**(1/3.)
        # k-space grid spacing.
        self.kfac = 2*sc.pi/self.L
        # Fermi Wavevector (infinite system).
        self.kf = (3*self.pol*sc.pi**2*self.ne/self.L**3)**(1/3.)
        # Fermi energy (inifinite systems).
        self.ef = 0.5*self.kf**2
        # Single particle eigenvalues and corresponding kvectors
        (self.spval, self.kval) = self.sp_energies(self.kfac, self.ecut)
        # Compress single particle eigenvalues by degeneracy.
        self.deg_e  = self.compress_spval(self.spval)
        # Number of plane waves.
        self.M = len(self.spval)


    def sp_energies(self, kfac, ecut):
        ''' Calculate the allowed kvectors and resulting single particle eigenvalues
        which can fit in the sphere in kspace determined by ecut.

    Parameters
    ----------
    kfac : float
        kspace grid spacing.
    ecut : float
        energy cutoff.

    Returns
    -------
    spval : list
        List containing single particle eigenvalues.
    kval : list
        List containing basis vectors.
    '''

        # Scaled Units to match with HANDE.
        # So ecut is measured in units of 1/kfac^2.
        nmax = int(math.ceil(np.sqrt((2*ecut))))

        spval = []
        vec = []
        kval = []

        for ni in range(-nmax, nmax+1):
            for nj in range(-nmax, nmax+1):
                for nk in range(-nmax, nmax+1):
                    spe = 0.5*(ni**2 + nj**2 + nk**2)
                    if (spe <= ecut):
                        kval.append([ni,nj,nk])
                        # Reintroduce 2 \pi / L factor.
                        spval.append(kfac**2*spe)

        # Sort the arrays in terms of increasing energy.
        spval = np.array(spval)
        kval = [x for y, x in sorted(zip(spval, kval))]
        kval = np.array(kval)
        spval.sort()

        return (spval, kval)


    def compress_spval(self, spval):
        ''' Compress the single particle eigenvalues so that we only consider unique
        values which vastly speeds up the k-space summations required.

    Parameters
    ----------
    spval : list
        list containing single particle eigenvalues
    Returns
    -------
    def_e : list of lists
        Compressed single-particle eigenvalues.
    '''

        # Work out the degeneracy of each eigenvalue.
        j = 1
        i = 0
        it = 0
        deg_e = []

        while it < len(spval)-1:
            eval1 = spval[i]
            eval2 = spval[i+j]
            #print abs(eval2-eval1), eval1, eval3
            if np.abs(eval2-eval1) < 1e-12:
                j += 1
            else:
                deg_e.append([j,eval1])
                i += j
                j = 1
            it += 1

        deg_e.append([j,eval1])

        return deg_e
