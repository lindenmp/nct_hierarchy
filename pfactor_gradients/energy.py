import numpy as np
import scipy as sp
import math
from scipy.linalg import svd
from tqdm import tqdm

# %% functions
def matrix_normalization(A, version=None, c=1, verbose=False):
    '''

    Args:
        A: np.array (n_parcels, n_parcels)
            adjacency matrix from structural connectome
        version: str
            options: 'continuous' or 'discrete'. default=None
            string variable that determines whether A is normalized for a continuous-time system or a discrete-time
            system. If normalizing for a continuous-time system, the identity matrix is subtracted.
        c: int
            normalization constant, default=1
    Returns:
        A_norm: np.array (n_parcels, n_parcels)
            normalized adjacency matrix

    '''

    if verbose:
        if version == 'continuous':
            print("Normalizing A for a continuous-time system")
        elif version == 'discrete':
            print("Normalizing A for a discrete-time system")
        elif version == None:
            raise Exception("Time system not specified. "
                            "Please nominate whether you are normalizing A for a continuous-time or a discrete-time system "
                            "(see function help).")

    # singluar value decomposition
    u, s, vt = svd(A)

    # Matrix normalization for discrete-time systems
    A_norm = A / (c + s[0])

    if version == 'continuous':
        # for continuous-time systems
        A_norm = A_norm - np.eye(A.shape[0])

    return A_norm


def minimum_energy(A, T, B, x0, xf):
    """
    :param A:
    :param T:
    :param B:
    :param x0:
    :param xf:
    :param c:
    :return:
    """
    #  Computes minimum control energy for state transition.
    #  A: System adjacency matrix:         N x N
    #  B: Control input matrix:            N x k
    #  x0: Initial state:                  N x 1
    #  xf: Final state:                    N x 1
    #  T: Control horizon                  1 x 1
    #
    #  Outputs
    #  x: State Trajectory
    #  u: Control Input

    # Author: Jennifer Stiso

    # System Size
    n = A.shape[0]  # Number of nodes

    if type(x0[0]) == np.bool_:
        x0 = x0.astype(float)
    if x0.ndim == 1:
        x0 = x0.reshape(-1, 1)

    if type(xf[0]) == np.bool_:
        xf = xf.astype(float)
    if xf.ndim == 1:
        xf = xf.reshape(-1, 1)

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


def expand_states(states):

    unique, counts = np.unique(states, return_counts=True)
    n_parcels = len(states)
    n_states = len(unique)

    x0_mat = np.zeros((n_parcels, 1)).astype(bool)
    xf_mat = np.zeros((n_parcels, 1)).astype(bool)

    for i in np.arange(n_states):
        for j in np.arange(n_states):
            x0 = states == i
            xf = states == j

            x0_mat = np.append(x0_mat, x0.reshape(-1, 1), axis=1)
            xf_mat = np.append(xf_mat, xf.reshape(-1, 1), axis=1)

    x0_mat = x0_mat[:, 1:]
    xf_mat = xf_mat[:, 1:]

    return x0_mat, xf_mat


def minimum_energy_fast(A, T, B, x0, xf, return_regional=False):
    # System Size
    n_parcels = A.shape[0]

    try:
        if type(x0[0][0]) == np.bool_:
            x0 = x0.astype(float)
        if type(xf[0][0]) == np.bool_:
            xf = xf.astype(float)
    except:
        if type(x0[0]) == np.bool_:
            x0 = x0.astype(float)
        if type(xf[0]) == np.bool_:
            xf = xf.astype(float)

    if x0.ndim == 1:
        x0 = x0.reshape(-1, 1)
    if xf.ndim == 1:
        xf = xf.reshape(-1, 1)

    # Number of integration steps
    nt = 1000
    dt = T/nt

    # Numerical integration with Simpson's 1/3 rule
    # Integration step
    dE = sp.linalg.expm(A * dt)
    # Accumulation of expm(A * dt)
    dEA = np.eye(n_parcels)
    # Gramian
    G = np.zeros((n_parcels, n_parcels))

    for i in np.arange(1, nt/2):
        # Add odd terms
        dEA = np.matmul(dEA, dE)
        p1 = np.matmul(dEA, B)
        # Add even terms
        dEA = np.matmul(dEA, dE)
        p2 = np.matmul(dEA, B)
        G = G + 4 * (np.matmul(p1, p1.transpose())) + 2 * (np.matmul(p2, p2.transpose()))

    # Add final odd term
    dEA = np.matmul(dEA, dE)
    p1 = np.matmul(dEA, B)
    G = G + 4 * (np.matmul(p1, p1.transpose()))

    # Divide by integration step
    E = sp.linalg.expm(A * T)
    G = (G + np.matmul(B, B.transpose()) + np.matmul(np.matmul(E, B), np.matmul(E, B).transpose())) * dt / 3
    # G = np.eye(n_parcels)

    delx = xf - np.matmul(E, x0)
    if return_regional:
        E = np.multiply(np.matmul(np.linalg.pinv(G), delx), delx)
    else:
        E = np.sum(np.multiply(np.matmul(np.linalg.pinv(G), delx), delx), axis=0)

    return E


def control_energy_helper(A, states, B, T=1, control='minimum_fast'):

    unique, counts = np.unique(states, return_counts=True)
    n_states = len(unique)

    E = np.zeros((n_states, n_states))
    n_err = np.zeros((n_states, n_states))

    # create state space by expanding state vector to matrices
    x0_mat, xf_mat = expand_states(states=states)

    # run control code
    if control == 'minimum_fast':
        e = minimum_energy_fast(A, T, B, x0_mat, xf_mat)
        E[:] = e.reshape(n_states, n_states)
        n_err[:] = np.nan
    elif control == 'minimum':
        col = 0
        for i in np.arange(n_states):
            for j in np.arange(n_states):

                x, u, n_err[i, j] = minimum_energy(A, T, B, x0_mat[:, col], xf_mat[:, col])
                E[i, j] = np.sum(np.square(u))

                col += 1

    return E, n_err


def control_energy_brainmap(A, states, T=1, B='wb'):
    n_parcels = A.shape[0]
    B_store = B

    unique, counts = np.unique(states, return_counts=True)
    n_states = len(unique)

    # create state space by expanding state vector to matrices
    x0_mat, xf_mat = expand_states(states=states)
    # x0_mat[:] = False
    # xf_mat[:] = False

    if type(B_store) == str and B_store != 'wb':
        B_store = 'wb'

    # get B matrix, this will either be identity or a brain map, since these are the only that work here
    if type(B_store) == str and B_store == 'wb':
        B = np.eye(n_parcels)
    else:
        B = np.zeros((n_parcels, n_parcels))
        B[np.eye(n_parcels) == 1] = B_store + 1

    E = minimum_energy_fast(A, T, B, x0_mat, xf_mat, return_regional=True)
    E = E.reshape(n_parcels, n_states, n_states)
    E = np.moveaxis(E, 0, -1)

    return E


def get_gmat(A, T=1):
    # System Size
    n_parcels = A.shape[0]

    # Gradient precalculations
    gmat = np.zeros((n_parcels, n_parcels, n_parcels))

    # Simpson's integration
    nt = 1000
    dt = T / nt
    dE = sp.linalg.expm(A * dt)
    dEA = np.zeros((n_parcels, n_parcels, nt+1))
    dEA[:, :, 0] = np.eye(n_parcels)

    for i in np.arange(1, nt+1):
        dEA[:, :, i] = np.matmul(dEA[:, :, i-1], dE)

    # Compute Gradient
    end = dEA.shape[2]
    for i in np.arange(0, n_parcels):
        dEAOdd = np.multiply(dEA[:, :, np.arange(1, end, 2)],
                             np.repeat(dEA[i, :, :][:, np.arange(1, end, 2)][np.newaxis, :, :], n_parcels, axis=0))

        dEAEven = np.multiply(dEA[:, :, np.arange(2, end - 1, 2)],
                              np.repeat(dEA[i, :, :][:, np.arange(2, end - 1, 2)][np.newaxis, :, :], n_parcels, axis=0))

        dEAOdd = np.sum(dEAOdd, axis=2)
        dEAEven = np.sum(dEAEven, axis=2)
        gmat[:, i, :] = 4 * dEAOdd \
                        + 2 * dEAEven \
                        + np.multiply(dEA[:, :, 0], dEA[i, :, 0]) \
                        + np.multiply(dEA[:, :, -1], dEA[i, :, -1])
        gmat[:, i, :] = gmat[:, i, :] * dt / 3

    return gmat


def grad_descent_b(A, B0, x0_mat, xf_mat, gmat, n=1, ds=0.1, T=1):
    # System Size
    n_parcels = A.shape[0]
    k = x0_mat.shape[1]

    V = np.matmul(sp.linalg.expm(A * T), x0_mat.astype(float)) - xf_mat.astype(float)
    # B_opt = B0.copy()
    B_opt = np.zeros((n_parcels, k, n + 1))
    B_opt[:, :, 0] = B0.copy()
    E_opt = np.zeros((k, n))

    # Iterate across state transitions
    for i in tqdm(np.arange(k)):
        # Iterate across gradient steps
        for j in np.arange(n):
            BM = B_opt[:, i, j].copy().reshape(1, 1, n_parcels)
            # Compute Gramian
            Wc = np.sum(np.multiply(gmat, BM ** 2), axis=2)
            vWcI = sp.linalg.solve(Wc, V[:, i].reshape(-1, 1))
            x1 = np.multiply(gmat, BM)
            x2 = np.multiply(x1, np.repeat(np.repeat(vWcI.transpose(), n_parcels, axis=0)[:, :, np.newaxis],
                                           n_parcels, axis=2))
            x3 = np.multiply(x2, np.repeat(np.repeat(vWcI, n_parcels, axis=1)[:, :, np.newaxis], n_parcels, axis=2))
            grad = np.sum(np.sum(x3, axis=0), axis=0)

            # get energy
            # Wc = np.sum(np.multiply(gmat, B_opt[:, i].reshape(1, 1, n_parcels) ** 2), axis=2)
            # vWcI = sp.linalg.solve(Wc, V[:, i].reshape(-1, 1))
            E_opt[i, j] = np.matmul(V[:, i].reshape(-1, 1).transpose(), vWcI)

            # update weights
            B_opt[:, i, j + 1] = B_opt[:, i, j] + grad / sp.linalg.norm(grad) * ds

    return B_opt, E_opt
