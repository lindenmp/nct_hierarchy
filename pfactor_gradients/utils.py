import numpy as np
import pandas as pd
from statsmodels.stats import multitest

def get_pdist_clusters(coords, labels, method='mean'):
    unique, counts = np.unique(labels, return_counts=True)
    n_clusters = len(unique)

    dist = np.zeros((n_clusters, n_clusters))

    for i in np.arange(n_clusters):
        for j in np.arange(n_clusters):
            x0 = labels == i
            xf = labels == j

            x0_coords = coords[x0, :]
            xf_coords = coords[xf, :]

            tmp = []
            for r1 in np.arange(x0_coords.shape[0]):
                for r2 in np.arange(xf_coords.shape[0]):
                    d = (x0_coords[r1, :] - xf_coords[r2, :]) ** 2
                    d = np.sum(d)
                    d = np.sqrt(d)
                    tmp.append(d)

            if method == 'mean':
                dist[i, j] = np.mean(tmp)
            elif method == 'min':
                dist[i, j] = np.min(tmp)
            elif method == 'median':
                dist[i, j] = np.median(tmp)

    return dist


def get_disc_repl(df, frac=0.5):
    df_out = pd.Series(data=np.zeros(df.shape[0], dtype=bool), index=df.index)

    n = np.round(df_out.shape[0] * frac).astype(int)
    hold_out = df_out.sample(n=n, random_state=0, replace=False, axis=0).index
    df_out[hold_out] = True
    print('Train:', np.sum(df_out == 0), 'Test:', np.sum(df_out == 1))

    return df_out


def mean_over_clusters(x, cluster_labels):
    unique = np.unique(cluster_labels, return_counts=False)
    n_clusters = len(unique)

    x_out = np.zeros((n_clusters, n_clusters))
    for i in np.arange(n_clusters):
        for j in np.arange(n_clusters):
            x_out[i, j] = np.nanmean(np.nanmean(x[cluster_labels == i, :], axis=0)[cluster_labels == j])

    return x_out


def get_exact_p(x, y):
    pval = 2 * np.min([np.mean(x - y >= 0), np.mean(x - y <= 0)])

    return pval


def get_null_p(E, E_null):
    return np.sum(E >= E_null) / len(E_null)


def get_fdr_p(p_vals, alpha=0.05):
    if p_vals.ndim == 2:
        dims = p_vals.shape
        p_vals = p_vals.reshape(-1)

        do_reshape = True

    out = multitest.multipletests(p_vals, alpha=alpha, method='fdr_bh')
    p_fdr = out[1]

    if do_reshape:
        p_fdr = p_fdr.reshape(dims)

    return p_fdr
