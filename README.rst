UEGPY
=====

UEGPY is a collection of python module which can be combined to calculate various
properties of the uniform electron gas (UEG). This was intended mostly for my own
education but can evaluate useful things albeit in an incredibly suboptimal fashion.

Currently we can do:

* Internal, potential and free energies for the the 3D-UEG as a function of
temperature at various levels of approximation. Currently we can do ideal,
Hartree--Fock and RPA.
* Dynamic properties such as density-density reposonse at any temperature.
* Static and dynamic structure factors.

Most of these quantities can be calculated in the thermodynamic limit as well as
for a finite number of electrons in the canonical and grand canonical ensemble.

This is a work in progress and probably full of bugs.

Documentation
-------------
Documentation is mostly in the form of function docstrings and is on `readthedocs
<https://uegpy.readthedocs.org>`_.

.. image:: http://readthedocs.org/projects/uegpy/badge/?version=latest
    :target: http://uegpy.readthedocs.io/en/latest/?badge=latest

LICENSE
-------
GPL v3.0
