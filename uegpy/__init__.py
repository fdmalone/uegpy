'''uegpy can calculate various thermodynamic properties of the UEG.

The code is organised in various modules which can be combined to calculate
useful quantities.  Currently the following functionality is available:

* Internal, potential and free energies for the the 3D-UEG as a function of
temperature at various levels of approximation. Currently we can do ideal,
Hartree--Fock and RPA.

* Dynamic properties such as density-density reposonse at any temperature.

* Static and dynamic structure factors.

Most of these quantities can be calculated in the thermodynamic limit as well as
for a finite number of electrons in the canonical and grand canonical ensemble.
'''
import uegpy.dielectric
import uegpy.utils
import uegpy.finite
import uegpy.infinite
import uegpy.monte_carlo
import uegpy.size_corrections
import uegpy.structure
import uegpy.ueg_sys
