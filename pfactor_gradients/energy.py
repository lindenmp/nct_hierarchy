import numpy as np
import scipy as sp
import math
from scipy.linalg import svd
from tqdm import tqdm

# %% functions
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


def optimal_energy(A, T, B, x0, xf, rho, S, c=1):
    """
    :param A:
    :param T:
    :param B:
    :param x0:
    :param xf:
    :param rho:
    :param S:
    :param c:
    :return:
    """
    # This is a python adaptation of matlab code originally written by Tomaso Menara and Jason Kim
    # % compute optimal inputs/trajectories
    # % Fabio, Tommy September 2017
    # %
    # % -------------- Change Log -------------
    # % JStiso April 2018
    # %   Changed S to be an input, rather than something defined internally
    # %
    # % Jason Kim January 2021
    # %   Changed the forward propagation of states to matrix exponential to
    # %   avoid reliance on MATLAB toolboxes. Also changed definition of expanded
    # %   input U to save time by avoiding having to resize the matrix.
    # %   Also changed the initialization of U_opt for the same reason.
    #
    # JStiso 2021
    #     Translated to Python

    # % Inputs:
    # % A     (NxN numpy array) Structural connectivity matrix
    # % B     (NxN numpy array) Input matrix: selects which nodes to put input into. Define
    # %       so there is a 1 on the diagonal of elements you want to add input to,
    # %       and 0 otherwise
    # % S     (NxN numpy array) Selects nodes whose distance you want to constrain, Define so
    # %       that there is a 1 on the diagonal of elements you want to
    # %       constrain, and a zero otherwise
    # % T     (float) Time horizon: how long you want to control for. Too large will give
    # %       large error, too short will not give enough time for control
    # % rho   (float) weights energy and distance constraints. Small rho leads to larger
    # %       energy
    #
    # Outputs:
    # X_opt    (TxN numpy array) The optimal trajectory through state space
    # U_opt    (TxN numpy array) The optimal energy
    # n_err    (float) the error associated with this calculation. Errors will be larger when B is not identity,
    #          and when A is large. Large T and rho will also tend to increase the error

    n = A.shape[0]  # Number of nodes

    u, s, vt = svd(A)  # singluar value decomposition
    A = A / (c + s[0]) - np.eye(A.shape[0])  # Matrix normalization

    if type(x0[0]) == np.bool_:
        x0 = x0.astype(float)
    if x0.ndim == 1:
        x0 = x0.reshape(-1, 1)

    if type(xf[0]) == np.bool_:
        xf = xf.astype(float)
    if xf.ndim == 1:
        xf = xf.reshape(-1, 1)

    Sbar = np.eye(n) - S
    np.shape(np.dot(-B, B.T) / (2 * rho))

    Atilde = np.concatenate((np.concatenate((A, np.dot(-B, B.T) / (2 * rho)), axis=1),
                             np.concatenate((-2 * S, -A.T), axis=1)), axis=0)

    M = sp.linalg.expm(Atilde * T)
    M11 = M[0:n, 0:n]
    M12 = M[0:n, n:]
    M21 = M[n:, 0:n]
    M22 = M[n:, n:]

    N = np.linalg.solve(Atilde, (M - np.eye(np.shape(Atilde)[0])))
    c = np.dot(np.dot(N, np.concatenate((np.zeros((n, n)), S), axis=0)), 2 * xf)
    c1 = c[0:n]
    c2 = c[n:]

    p0 = np.dot(np.linalg.pinv(np.concatenate((np.dot(S, M12), np.dot(Sbar, M22)), axis=0)),
                (-np.dot(np.concatenate((np.dot(S, M11), np.dot(Sbar, M21)), axis=0), x0) -
                 np.concatenate((np.dot(S, c1), np.dot(Sbar, c2)), axis=0) +
                 np.concatenate((np.dot(S, xf), np.zeros((n, 1))), axis=0)))

    n_err = np.linalg.norm(np.dot(np.concatenate((np.dot(S, M12), np.dot(Sbar, M22)), axis=0), p0) -
                           (-np.dot(np.concatenate((np.dot(S, M11), np.dot(Sbar, M21)), axis=0), x0) -
                            np.concatenate((np.dot(S, c1), np.dot(Sbar, c2)), axis=0) +
                            np.concatenate((np.dot(S, xf), np.zeros((n, 1))), axis=0)))  # norm(error)

    STEP = 0.001
    t = np.arange(0, (T + STEP), STEP)

    U = np.dot(np.ones((np.size(t), 1)), 2 * xf.T)

    # Discretize continuous-time input for convolution
    Atilde_d = sp.linalg.expm(Atilde * STEP)
    Btilde_d = np.linalg.solve(Atilde,
                               np.dot((Atilde_d - np.eye(2 * n)), np.concatenate((np.zeros((n, n)), S), axis=0)))

    # Propagate forward discretized model
    xp = np.zeros((2 * n, np.size(t)))
    xp[:, 0:1] = np.concatenate((x0, p0), axis=0)
    for i in np.arange(1, np.size(t)):
        xp[:, i] = np.dot(Atilde_d, xp[:, i - 1]) + np.dot(Btilde_d, U[i - 1, :].T)

    xp = xp.T

    U_opt = np.zeros((np.size(t), np.shape(B)[1]))
    for i in range(np.size(t)):
        U_opt[i, :] = -(1 / (2 * rho)) * np.dot(B.T, xp[i, n:].T)

    X_opt = xp[:, 0:n]

    return X_opt, U_opt, n_err


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


def minimum_energy_nonh(A, T, B, x0, xf, c=1):
    u, s, vt = svd(A)  # singluar value decomposition
    A = A / (c + s[0]) - np.eye(A.shape[0])  # Matrix normalization

    Phi = np.dot(sp.linalg.expm(A * T), x0) - xf

    Wc = np.zeros(A.shape)
    nN = 1000
    for i in np.arange(0, nN):
        FM = np.matmul(sp.linalg.expm(A * T * (i / nN)), B)
        Wc += np.matmul(FM, FM.T) * (T / nN)

    E = np.dot(Phi.T, sp.linalg.solve(Wc, Phi))

    return E


def get_time_vec(T, n_taylor):
    return np.power(T, np.arange(0, n_taylor))


def minimum_energy_taylor(A, T, B, x0, xf, c=1, n_taylor=10, drop_taylor=0):
    num_parcels = A.shape[0]  # Number of nodes

    # Normalize and eigendecompose
    u, s, vt = np.linalg.svd(A)  # singluar value decomposition
    A = A / (c + s[0]) - np.eye(A.shape[0])  # Matrix normalization
    w, v = np.linalg.eig(A)  # Eigenvalue decomposition

    # Define eigenvalue powers (~.25)
    w = np.reshape(w, (num_parcels, 1))
    p = np.reshape(np.arange(0, n_taylor), (1, n_taylor))
    W = np.power(w, p)

    # Define Taylor series coefficients
    tc = np.zeros(n_taylor)
    for i in np.arange(0, n_taylor):
        tc[i] = 1 / math.factorial(i)
    if drop_taylor > 0:
        tc[drop_taylor] = 0
    tc = np.reshape(tc, (1, n_taylor))

    # Multiple eigenvalues with coefficients
    W = np.multiply(W, tc)

    # Define time matrix
    nN = 1000
    t_mat = np.zeros((1, n_taylor, nN))
    for i in np.arange(0, nN):
        t_mat[0, :, i] = get_time_vec((T / nN) * i, n_taylor)

    # Perform numerical integration (~.25)
    WT = np.dot(W, t_mat)
    WT = np.multiply(WT, np.reshape(WT, (1, num_parcels, nN)))

    # Define Gramian
    P = np.matmul(v.T, B)
    P = np.matmul(P, P.T)
    P = np.reshape(P, (num_parcels, num_parcels, 1))
    WcM = np.multiply(WT, P)
    Wc = np.matmul(np.matmul(v, np.sum(WcM, 2)), v.T) * (T / nN)

    # State transition
    WPr = np.multiply(W, np.reshape(get_time_vec(T, n_taylor), (1, n_taylor)))
    EV = np.matmul(v, np.matmul(np.diag(np.sum(WPr, 1)), v.T))
    Phi = np.dot(EV, x0) - xf

    E = np.dot(Phi.T, np.linalg.solve(Wc, Phi))

    return E


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


def minimum_energy_fast(A, T, B, x0, xf, c=1, return_regional=False):
    # System Size
    n_parcels = A.shape[0]

    # singluar value decomposition
    u, s, vt = svd(A)
    # Matrix normalization
    A = A / (c + s[0]) - np.eye(n_parcels)

    if type(x0[0][0]) == np.bool_:
        x0 = x0.astype(float)
    if type(xf[0][0]) == np.bool_:
        xf = xf.astype(float)

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

    delx = xf - np.matmul(E, x0)
    if return_regional:
        E = np.multiply(np.matmul(np.linalg.pinv(G), delx), delx)
    else:
        E = np.sum(np.multiply(np.matmul(np.linalg.pinv(G), delx), delx), axis=0)

    return E


def control_energy_helper(A, states, n_subsamples=0, control='minimum_fast', T=1, B='wb', add_noise=False):
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

    # create state space by expanding state vector to matrices
    x0_mat, xf_mat = expand_states(states=states)
    if n_subsamples > 0:
        x0_sub = np.zeros((x0_mat.shape[0], x0_mat.shape[1], n_subsamples)).astype(bool)
        xf_sub = np.zeros((xf_mat.shape[0], xf_mat.shape[1], n_subsamples)).astype(bool)

        for j in np.arange(x0_mat.shape[1]):
            np.random.seed(0)
            for k in np.arange(n_subsamples):
                x0_sub[:, j, k] = subsample_state(x0_mat[:, j], subsample_size)
                xf_sub[:, j, k] = subsample_state(xf_mat[:, j], subsample_size)

    # run control code
    if control == 'minimum_fast':
        if type(B_store) == str and B_store != 'wb':
            B_store = 'wb'

        n_err[:] = np.nan

        # get B matrix, this will either be identity or a brain map, since these are the only that work here
        if type(B_store) == str and B_store == 'wb':
            B = np.eye(n_parcels)
        else:
            B = np.zeros((n_parcels, n_parcels))
            B[np.eye(n_parcels) == 1] = B_store + 1

        if add_noise:
            B[np.eye(n_parcels) == 1] = B[np.eye(n_parcels) == 1] + noise

        if n_subsamples > 0:
            for k in np.arange(n_subsamples):
                e = minimum_energy_fast(A, T, B, x0_sub[:, :, k], xf_sub[:, :, k])
                E[:, :, k] = e.reshape(n_states, n_states)
        else:
            e = minimum_energy_fast(A, T, B, x0_mat, xf_mat)
            E[:] = e.reshape(n_states, n_states)
    else:
        col = 0
        for i in np.arange(n_states):
            for j in np.arange(n_states):
                if n_subsamples > 0:
                    for k in np.arange(n_subsamples):
                        B = get_B_matrix(x0_sub[:, col, k], xf_sub[:, col, k], version=B_store)
                        if add_noise:
                            B[np.eye(n_parcels) == 1] = B[np.eye(n_parcels) == 1] + noise

                        if control == 'minimum':
                            x, u, n_err[i, j, k] = minimum_energy(A, T, B, x0_sub[:, col, k], xf_sub[:, col, k])
                            E[i, j, k] = np.sum(np.square(u))
                else:
                    B = get_B_matrix(x0_mat[:, col], xf_mat[:, col], version=B_store)
                    if add_noise:
                        B[np.eye(n_parcels) == 1] = B[np.eye(n_parcels) == 1] + noise

                    if control == 'minimum':
                        x, u, n_err[i, j] = minimum_energy(A, T, B, x0_mat[:, col], xf_mat[:, col])
                        E[i, j] = np.sum(np.square(u))

                col += 1

    if n_subsamples > 0:
        E = np.nanmean(E, axis=2)
        n_err = np.nanmean(n_err, axis=2)

    return E, n_err


def control_energy_brainmap(A, states, T=1, B='wb'):
    n_parcels = A.shape[0]
    B_store = B

    unique, counts = np.unique(states, return_counts=True)
    n_states = len(unique)

    # create state space by expanding state vector to matrices
    x0_mat, xf_mat = expand_states(states=states)

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