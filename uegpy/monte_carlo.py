
def sample_canonical_energy(system, beta, mu, nmeasure):

    p_i = np.array([fermi_factor(ek, mu, beta) for ek in system.spval])
    evals = system.spval
    E = np.zeros(nmeasure)

    en = 0
    Z = 0

    for it in range(0,nmeasure):

        (gen, orb_list) = create_orb_list(p_i, system.ne, system.M)
        if gen:
            E[Z] = sum(evals[orb_list])
            Z += 1

        if it % 1000 == 0 and Z > 0:
            print(it, np.mean(E[:Z]), np.std(E[:Z], ddof=1)/np.sqrt(Z))



def create_orb_list(probs, ne, M):

    selected_orbs = np.zeros(ne, dtype=np.int)
    nselect = 0

    for iorb in range(0,M):

        r = rand.random()

        if (probs[iorb] > r):
            nselect += 1
            if (nselect > ne):
                gen = False
                break
            selected_orbs[nselect-1] = iorb

    if (nselect == ne):
        gen = True
    else:
        gen = False

    return (gen, selected_orbs)

def gc_correction_free_energy(sys, cpot, beta, delta, delta_error):

    # Assumption, Z_GC(N) / Z_GC = naccept/ntotal = delta, so -kT log(delta) = -kT log Z_GC(N) + kT log Z_GC
    # So -kT log Z_N = -kT log(delta) - kT log(Z_GC) + mu N, or F^0_N = F^0_GC + Delta(N) + mu N.

    muN = cpot * sys.ne
    F_GC = -(1.0/beta)*(np.log(gc_part_func(sys, cpot, beta)))
    Delta = -1.0/beta*np.log(delta)

    DF_N = F_GC + muN
    F_N = DF_N + Delta
    F_N_error = delta_error / (beta*delta)

    return (F_N, F_N_error, DF_N, F_GC)
