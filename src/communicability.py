import numpy as np
from bct.algorithms.distance import distance_wei_floyd, retrieve_shortest_path
from tqdm import tqdm


def matching_index(W):
    n_nodes = W.shape[0]
    m = np.zeros((n_nodes, n_nodes))

    for i in np.arange(n_nodes - 1):
        for j in np.arange(i + 1, n_nodes):
            x, y, z, = 0, 0, 0
            for k in np.arange(n_nodes):
                if W[i, k] != 0 and W[j, k] != 0 and k != i and k != j:
                    x = x + W[i, k] + W[j, k]

                if k != j:
                    y = y + W[i, k]

                if k != i:
                    z = z + W[j, k]

            m[i, j] = x / (y + z)

    m = m + m.transpose()

    return m


def get_pt(m, hops, Pmat):
    n_nodes = hops.shape[0]
    pt = np.zeros((n_nodes, n_nodes))

    for i in np.arange(n_nodes - 1):
        for j in np.arange(i + 1, n_nodes):
            x = 0
            path = retrieve_shortest_path(i, j, hops, Pmat)
            K = len(path)

            for t in np.arange(K - 1):
                for l in np.arange(t + 1, K):
                    x = x + m[path[t], path[l]]

            pt[i, j] = 2 * x / (K * (K - 1))

    pt = pt + pt.transpose()

    return pt


def path_transitivity(W, transform='inv'):
    m = matching_index(W)
    D, hops, Pmat = distance_wei_floyd(W, transform=transform)
    pt = get_pt(m, hops, Pmat)

    return pt


def get_pt_cum(m, path):
    K = len(path)

    pt_ij = []
    pt_ji = []

    for i in np.arange(2, K + 1):
        sub_path = path[:i]
        K_sub = len(sub_path)

        x = 0
        for j in np.arange(K_sub - 1):
            for k in np.arange(j + 1, K_sub):
                x = x + m[sub_path[j], sub_path[k]]
        # t = 2 * x / (K_sub * (K_sub - 1))
        t = 2 * x / (K * (K - 1))
        pt_ij.append(t[0])

        # get reverse direction, j to i
        sub_path = np.flip(path)[:i]
        K_sub = len(sub_path)

        x = 0
        for j in np.arange(K_sub - 1):
            for k in np.arange(j + 1, K_sub):
                x = x + m[sub_path[j], sub_path[k]]
        # t = 2 * x / (K_sub * (K_sub - 1))
        t = 2 * x / (K * (K - 1))
        pt_ji.append(t[0])

    return pt_ij, pt_ji


def cumulative_transitivity_differences(m, hops, Pmat):
    n_nodes = hops.shape[0]
    pt_cd = np.zeros((n_nodes, n_nodes))

    for i in tqdm(np.arange(n_nodes)):
        for j in np.arange(n_nodes):
            path = retrieve_shortest_path(i, j, hops, Pmat)
            pt_ij, pt_ji = get_pt_cum(m, path)

            pt_cd[i, j] = np.sum(np.asarray(pt_ij) - np.asarray(pt_ji))

    return pt_cd
