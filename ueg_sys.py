''' System properties.
'''

import scipy as sc
import numpy as np
import math

class System:
    ''' System Properties - Everything is measured in Hartree atomic units.
    rs: seitz radius.
    ne: number of electrons.
    ecut: plane wave cutoff.
    pol: spin polarisation = 2 for fully polarised case.
    L: box length.
    kfac: kspace grid spacing.
    kf: Fermi wavevector.
    ef: Fermi energy
    integeral_factor: common factor for all Fermi integrals.
    spval: array containing single particle eigenvalues, sorted in increasing value of kinetic energy.
    kval: plane wave basis vectors, sorted in increasing value of kinetic energy.
    deg_e: compressed list of spval containing unique eigenvalues and their degeneracy.
    deg_k: REMOVE.
    M: number of plane waves in our basis.
    energy_fin: total energy for ne electrons in a finite box.
    energy_inf: total energy for ne electrons in an inifinite box.
    total_K: symmetry of determinants being considered.
    beta_max: maximum beta value which we calculated properties to.
    ef_fin: Fermi energy for finite system.
'''

    def __init__(self, args):

        # Seitz radius.
        self.rs = float(args[0])
        # Number of electrons.
        self.ne = int(args[1])
        # Kinetic energy cut-off.
        self.ecut = float(args[2])
        # Spin polarisation.
        self.pol = int(args[3])
        # Box Length.
        self.L = self.rs*(4*self.ne*sc.pi/3.)**(1/3.)
        # k-space grid spacing.
        self.kfac = 2*sc.pi/self.L
        # Fermi Wavevector (infinite system).
        self.kf = (3*self.pol*sc.pi**2*self.ne/self.L**3)**(1/3.)
        # Fermi energy (inifinite systems).
        self.ef = 0.5*self.kf**2
        # Integral Factor.
        self.integral_factor = self.L**3 * np.sqrt(2) / (self.pol*sc.pi**2)
        ## Single particle eigenvalues and corresponding kvectors
        (self.spval, self.kval) = self.sp_energies(self.kfac, self.ecut)
        # Compress single particle eigenvalues by degeneracy.
        (self.deg_e, self.deg_k) = self.compress_spval(self, self.spval, self.kval)
        # Number of plane waves.
        self.M = len(self.spval)
        (self.t_energy_fin, self.t_energy_inf) = self.total_energy_T0(self, self.spval, self.ef, self.ne)
        self.ef_fin = self.dis_fermi(self, self.spval, self.ne)
        # Change this to be more general, selected to be the gamma point.
        self.total_K = self.kval[0]
        self.beta_max = 5.0

    def print_system_variables(self):
        ''' Print out system varaibles.'''

        print " # Number of electrons: ", self.ne
        print " # Spin polarisation: ", self.pol
        print " # rs: ", self.rs
        print " # System Length: ", self.L
        print " # Fermi wavevector for infinite system: ", self.kf
        print " # kspace grid spacing: ", self.kfac
        print " # Number of plane waves: ", self.spval.size
        print " # Fermi energy for infinite system: ", self.ef
        print " # Fermi energy for discrete system: ", self.ef_fin
        print " # Ground state energy for finite system: ", self.t_energy_fin
        print " # Ground state total energy for infinite system: ", self.t_energy_inf
        self.root_de = 1e-10

    # Remove these.
    def kf(self, ne, L, ns):

        k_f = (3*ns*sc.pi**2*ne/L**3)**(1/3.0)

        return k_f
    # Remove.
    def ef(self, ne, L, ns):

        e_f = 0.5*(3*ns*sc.pi**2*ne/L**3)**(2/3.)

        return e_f
    # Remove.
    def setup(self, rs, ne):

        # System Lenght.
        L = rs*(4*ne*sc.pi/3.)**(1/3.)
        # k = 2*pi/L n
        kfac = 2*sc.pi/L

        return (L, kfac)
    # Remove.
    def total_energy_T0(self, spval, E_f, ne):

        return (sum(spval[:ne]), 3*ne/5.*E_f)

    def sp_energies(self, kfac, ecut):
        ''' Calculate the allowed kvectors and resulting single particle eigenvalues
        which can fit in the sphere in kspace determined by ecut.
    Params
    ------
    kfac: float
        kspace grid spacing.
    ecut: float
        energy cutoff.

    Returns
    -------
    spval: list
        list containing single particle eigenvalues
    kval: list
        list containing basis vectors.
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

    def compress_spval(self, spval, kval):
        ''' Compress the single particle eigenvalues so that we only consider unique
        values which vastly speeds up the k-space summations required.
        [todo] - Look at more clever optimisations (stars).

    Params
    ------
    spval: list
        list containing single particle eigenvalues
    kval: list
        list containing momentum space basis functions.
    '''

        # Work out the degeneracy of each eigenvalue.
        j = 1
        i = 0
        it = 0
        deg_e = []
        deg_k = []

        while it < len(spval)-1:
            eval1 = spval[i]
            eval2 = spval[i+j]
            if eval2 == eval1:
                j += 1
            else:
                deg_e.append([j,eval1])
                deg_k.append([j,kval[i]])
                i += j
                j = 1
            it += 1

        deg_e.append([j,eval1])
        deg_k.append([j,kval[i]])

        return (deg_e, deg_k)
