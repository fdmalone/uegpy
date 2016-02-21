''' Various approximate fits to the UEG or OCP '''


def classical_ocp(system, beta):
    ''' Evaluate the classical excess energy using the parametrised fit given by
    J. P. Hansen PRA 8, 6 1973.


Parameters
----------
system : class
    System being studied.
beta : float
    Inverse Temperature.

Returns
-------
U_xc : float
    Excess internal energy for classical OCP.

'''

    a1 = -0.895929
    b1 = 4.666486
    a2 = 0.113406
    b2 = 13.67541
    a3 = -0.908728
    b3 = 1.890560
    a4 = -0.116147
    b4 = 1.027755

    gamma = 1.0/(system.rs*T)
    U_xc = (1.5 * gamma**1.5 * (a1/(b1+gamma)**0.5 + a2/(b2+gamma)
            + a3/(b3+gamma)**1.5 + a4/(b4+gamma)**2))

    return U_xc
