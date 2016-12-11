import numpy as np
import scipy as sc
import ueg_sys as ue
import pandas as pd

class Basis:

    def __init__(self, x1, x2, x3):

        self.kvol = np.dot(x1, np.cross(x2, x3))

        self.b1 = self.reciprocal_lattice_vector(x2, x3, self.kvol)
        self.b2 = self.reciprocal_lattice_vector(x3, x1, self.kvol)
        self.b3 = self.reciprocal_lattice_vector(x1, x2, self.kvol)


    def reciprocal_lattice_vector(self, x1, x2, vol):

        return 2*sc.pi * np.cross(x1, x2) / vol



def kinetic_energy(k):

    return 0.5 * np.dot(k, k)


def potential_energy(G1, G2, positions, gvals, vlocal, volume):

    dG = G1 - G2
    struc = sum(np.cos(np.dot(dG, ratom)) for ratom in positions)

    return struc * np.interp(np.dot(dG,dG), gvals, vlocal)/ volume


def gaussian_potential(G1, G2, volume):

    dG = G1 - G2

    return np.exp((G1-G2)**2.0) / volume


def construct_hamil(system, atoms, gvals, vlocal, volume):

    H = np.zeros((len(system.kval), len(system.kval)))
    for (i, G1) in enumerate(system.kval):
        H[i, i] = system.kfac*kinetic_energy(G1)
        for (j, G2) in enumerate(system.kval):
            if i != j:
                H[i, j] = potential_energy(system.kfac*G1, system.kfac*G2, atoms, gvals, vlocal, volume)


    eigs = np.sort(np.linalg.eig(H)[0])

    return eigs


def read_pseudo(name):

    data = pd.read_csv(name, sep=r'\s+', skiprows=14)

    return (data.G.values, data.vloc.values)


def total_energy(a, nel, atoms, pseudo_file):

    rs = (3.0 / (4*sc.pi*nel))**(1.0/3.0) * a
    system = ue.System(rs, nel, 2, 0)
    (gvals, pseudo) = read_pseudo(pseudo_file)
    eigs = construct_hamil(system, atoms, gvals, pseudo, a**3.0)

    for i in range(0, 19):
        print eigs[i], system.spval[i]

    return sum(2*eigs[:nel/2]), sum(2*system.spval[:nel/2]), eigs[0], system.spval[0]
