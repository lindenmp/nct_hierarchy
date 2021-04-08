def get_B_matrix(x0, xf, version='wb'):
    n_parcels = x0.shape[0]

    if type(version) == str:
        if version == 'wb':
            B = np.eye(n_parcels)
        elif version == 'x0xf':
            B = np.zeros((n_parcels, n_parcels))
            B[x0, x0] = 1
            B[xf, xf] = 1
        elif version == 'x0':
            B = np.zeros((n_parcels, n_parcels))
            B[x0, x0] = 1
        elif version == 'xf':
            B = np.zeros((n_parcels, n_parcels))
            B[xf, xf] = 1
        elif version == 'x0xfwb':
            B = np.zeros((n_parcels, n_parcels))
            B[np.eye(n_parcels) == 1] = 5 * 10e-5
            B[x0, x0] = 1
            B[xf, xf] = 1
        elif version == 'x0wb':
            B = np.zeros((n_parcels, n_parcels))
            B[np.eye(n_parcels) == 1] = 5 * 10e-5
            B[x0, x0] = 1
        elif version == 'xfwb':
            B = np.zeros((n_parcels, n_parcels))
            B[np.eye(n_parcels) == 1] = 5 * 10e-5
            B[xf, xf] = 1
    else:
        B = np.zeros((n_parcels, n_parcels))
        B[np.eye(n_parcels) == 1] = version + 1

    return B


def subsample_state(x, subsample_size):
    x_tmp = np.zeros(x.size).astype(bool)

    sample = np.random.choice(np.where(x == True)[0], size=subsample_size, replace=False)

    x_tmp[sample] = True

    return x_tmp


def minimum_energy(A, T, B, x0, xf, c=1):
    """
    :param A:
    :param T:
    :param B:
    :param x0:
    :param xf:
    :param c:
    :return:
    """

    # Author: Jennifer Stiso

    # System Size
    n = A.shape[0]  # Number of nodes

    u, s, vt = svd(A)  # singluar value decomposition
    A = A / (c + s[0]) - np.eye(A.shape[0])  # Matrix normalization

    if type(x0[0]) == np.bool_:
        x0 = x0.astype(float)
    if type(xf[0]) == np.bool_:
        xf = xf.astype(float)

    # Compute Matrix Exponential
    AT = np.concatenate((np.concatenate((A, -.5 * (B.dot(B.T))), axis=1),
                         np.concatenate((np.zeros(np.shape(A)), -A.T), axis=1)), axis=0)

    E = sp.linalg.expm(AT * T)

    # Compute Costate Initial Condition
    E12 = E[0:n, n:]
    E11 = E[0:n, 0:n]
    p0 = np.linalg.pinv(E12).dot(xf - E11.dot(x0))

    # Compute Costate Initial Condition Error Induced by Inverse
    n_err = np.linalg.norm(E12.dot(p0) - (xf - E11.dot(x0)))

    # Prepare Simulation
    nStep = 1000
    t = np.linspace(0, T, nStep + 1)

    v0 = np.concatenate((x0, p0), axis=0)  # Initial Condition
    v = np.zeros((2 * n, len(t)))  # Trajectory
    Et = sp.linalg.expm(AT * T / (len(t) - 1))
    v[:, 0] = v0.T

    # Simulate State and Costate Trajectories
    for i in np.arange(1, len(t)):
        v[:, i] = Et.dot(v[:, i - 1])

    x = v[0:n, :];
    u = -0.5 * B.T.dot(v[np.arange(0, n) + n, :])

    # transpose to be similar to opt_eng_cont
    u = u.T
    x = x.T

    return x, u, n_err

def control_energy_helper(A, states, n_subsamples=0, T=1, B='wb', add_noise=False):
    n_parcels = A.shape[0]
    B_store = B

    unique, counts = np.unique(states, return_counts=True)
    n_states = len(unique)
    subsample_size = np.min(counts)

    if n_subsamples > 0:
        E = np.zeros((n_states, n_states, n_subsamples))
        n_err = np.zeros((n_states, n_states, n_subsamples))
    else:
        E = np.zeros((n_states, n_states))
        n_err = np.zeros((n_states, n_states))

    np.random.seed(0)
    if add_noise:
        noise = np.random.rand(n_parcels) * 0.1

    for i in tqdm(np.arange(n_states)):
        for j in np.arange(n_states):
            if i != j:
                np.random.seed(0)

                x0 = states == i
                xf = states == j

                if n_subsamples > 0:
                    for k in np.arange(n_subsamples):
                        x0_tmp = subsample_state(x0, subsample_size)
                        xf_tmp = subsample_state(xf, subsample_size)

                        B = get_B_matrix(x0_tmp, xf_tmp, version=B_store)
                        if add_noise:
                            B[np.eye(n_parcels) == 1] = B[np.eye(n_parcels) == 1] + noise

                        x, u, n_err[i, j, k] = minimum_energy(A, T, B, x0_tmp, xf_tmp)
                        E[i, j, k] = np.sum(np.square(u))
                else:
                    B = get_B_matrix(x0, xf, version=B_store)
                    if add_noise:
                        B[np.eye(n_parcels) == 1] = B[np.eye(n_parcels) == 1] + noise

                    x, u, n_err[i, j] = minimum_energy(A, T, B, x0, xf)
                    E[i, j] = np.sum(np.square(u))

    return E, n_err