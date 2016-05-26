import numpy as np
import scipy as sc

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
