import pandas as pd
import numpy as np
import observables as obs
import time

def run_calcs(system, calc='All'):
    '''Run user defined calculations.

Parameters
----------
system: class object
    System parameters
calc: list of strings
    What calculations to run on system

Returns
-------
data : pandas data frame containing desired quantities.
'''

    data = pd.DataFrame()
    beta = np.arange(0.1/system.ef, system.beta_max, 0.1/system.ef)
    data['Beta'] = beta
    data['T/T_F'] = 1.0 / (system.ef*beta)
    data['M'] = system.M
    calc_time = []

    if calc == 'All':
        start = time.time()
        # Find the chemical potential.
        mu = [obs.chem_pot_sum(system, b) for b in beta]
        data['chem_pot'] = mu
        # Evaluate observables.
        beta_mu = zip(beta, mu)
        data['Energy_sum'] = [obs.energy_sum(b, m, system.deg_e) for b, m in beta_mu]
        end = time.time()
        calc_time.append(end-start)
        start = time.time()
        mu = [obs.chem_pot_integral(system, b) for b in beta]
        beta_mu = zip(beta, mu)
        data['Energy_integral'] = [obs.energy_integral(b, m, system.integral_factor) for b, m in beta_mu]
        data['HFX_integral'] = [obs.hfx_integral(system, b, m) for b, m in beta_mu]
        end = time.time()
        calc_time.append(end-start)
        print " # Time taken for calculation: ", calc_time
    elif calc == 'partition':
        xval = np.arange(0,5,0.1)
        (tenergy, partition, count) = obs.canonical_partition_function(xval, system.spval, system.ne, system.kval, system.kval[0])
        data['Beta'] = xval
        data['Partition'] = tenergy / partition
    elif calc == 'classical':
        (T, Uxc) = obs.classical_ocp(system, 0.01, 10)
        data['T'] = T
        data['Theta'] = T / system.ef
        data['Beta'] = 1 / T
        data['Classical_Uxc'] = Uxc
    elif calc == 'com':
        data['Beta'] = beta
        data['E_COM'] = [obs.centre_of_mass_obsv(system, b)[0] for b in beta]

    return data
