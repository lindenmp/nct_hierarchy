import numpy as np
import scipy as sp
from scipy import stats
from scipy import signal
from bct.algorithms.distance import distance_wei_floyd, retrieve_shortest_path
from sklearn.linear_model import LinearRegression

class DataMatrix():
    def __init__(self, data=[]):
        self.data = data

    def get_distance_matrix(self):
        self.D, self.hops, self.Pmat = distance_wei_floyd(self.data, transform='inv')

    def regress_nuisance(self, c, indices=[]):
        if len(indices) == 0:
            # if no indices are given, filter out identity and nans
            indices = np.where(~np.eye(self.data.shape[0], dtype=bool) * ~np.isnan(self.data))

        x_out = np.zeros(self.data.shape)

        x = self.data[indices].reshape(-1, 1)
        c = c[indices].reshape(-1, 1)

        nuis_reg = LinearRegression()
        nuis_reg.fit(c, x)

        x_pred = nuis_reg.predict(c)
        x_out[indices] = x[:, 0] - x_pred[:, 0]

        self.data_resid = x_out

    def mean_over_clusters(self, cluster_labels, use_resid_matrix=False):
        if use_resid_matrix == True:
            try:
                x = self.data_resid
            except AttributeError:
                print('Warning: self.data_resid does not exist. run self.regress_nuisance() first and rerun '
                      'self.mean_over_clusters')
                print('..forcing self.mean_over_clusters(use_resid_matrix=False)')
                x = self.data
        elif use_resid_matrix == False:
            x = self.data

        unique = np.unique(cluster_labels, return_counts=False)
        n_clusters = len(unique)

        x_out = np.zeros((n_clusters, n_clusters))
        for i in np.arange(n_clusters):
            for j in np.arange(n_clusters):
                x_out[i, j] = np.nanmean(np.nanmean(x[cluster_labels == i, :], axis=0)[cluster_labels == j])

        self.data_clusters = x_out

    def check_disconnected_nodes(self):
        if np.any(np.sum(self.data, axis=1) == 0):
            self.disconnected_nodes = True
        else:
            self.disconnected_nodes = False

    def get_gradient_variance(self, gradients, return_abs=False):
        """
        :param gradients: (n_parcels,2) ndarray.
            Assumes transmodal gradient is on column 0 and unimodal gradient is on column 1
        :param return_abs: bool
            Whether variance along shortest paths is based on absolute differences or not

        :return:
            self.tm_var:
                variance of principal gradient traversal of shortest paths linking nodes i and j
            self.smv_var:
                variance of sensorimotor gradient traversal of shortest paths linking nodes i and j
            self.joint_var:
                variance along both gradients estimated via euclidean distance
        """
        print('Getting gradient variance over shortest paths...')
        try:
            self.hops
        except AttributeError:
            self.get_distance_matrix()

        n_parcels = self.data.shape[0]

        self.tm_var = np.zeros((n_parcels, n_parcels))
        self.smv_var = np.zeros((n_parcels, n_parcels))
        self.joint_var = np.zeros((n_parcels, n_parcels))

        for i in np.arange(n_parcels):
            for j in np.arange(n_parcels):
                if j > i:
                    shortest_path = retrieve_shortest_path(i, j, self.hops, self.Pmat)
                    if len(shortest_path) != 0:
                        gradient_diff = np.diff(gradients[shortest_path[:, 0], :], axis=0)

                        if return_abs == True:
                            gradient_diff = np.abs(gradient_diff)

                        # get the variance of the differences along the shortest path
                        var_diff = np.var(gradient_diff, axis=0)
                        self.tm_var[i, j] = var_diff[0]
                        self.smv_var[i, j] = var_diff[1]

                        # get the variance of the euclidean distance
                        self.joint_var[i, j] = np.var(np.sqrt(np.sum(np.square(gradient_diff), axis=1)))
                    else:
                        self.tm_var[i, j] = np.nan
                        self.smv_var[i, j] = np.nan
                        self.joint_var[i, j] = np.nan

        self.tm_var = self.tm_var + self.tm_var.transpose()
        self.smv_var = self.smv_var + self.smv_var.transpose()
        self.joint_var = self.joint_var + self.joint_var.transpose()

    def get_gene_coexpr_variance(self, gene_expression, return_abs=False):
        """
        :param gene_expression:
        :param return_abs:
        :return:
        """
        print('Getting gene coexpression mean/variance over shortest paths...')
        try:
            self.hops
        except AttributeError:
            self.get_distance_matrix()

        n_parcels = self.data.shape[0]

        self.gene_coexpr_mean = np.zeros((n_parcels, n_parcels))
        self.gene_coexpr_var = np.zeros((n_parcels, n_parcels))

        for i in np.arange(n_parcels):
            for j in np.arange(n_parcels):
                if j > i:
                    shortest_path = retrieve_shortest_path(i, j, self.hops, self.Pmat)
                    if len(shortest_path) != 0:
                        # get the coexpression for neighboring nodes along shortest_path
                        gene_coexpr = np.diag(np.corrcoef(gene_expression[shortest_path[:, 0], :], rowvar=True), k=1)

                        if return_abs == True:
                             gene_coexpr = np.abs(gene_coexpr)

                        # mean the coexpression along the shortest path
                        self.gene_coexpr_mean[i, j] = np.nanmean(gene_coexpr)
                        # get the variance of the coexpression along the shortest path
                        self.gene_coexpr_var[i, j] = np.nanvar(gene_coexpr)
                    else:
                        self.gene_coexpr_mean[i, j] = np.nan
                        self.gene_coexpr_var[i, j] = np.nan

        self.gene_coexpr_mean = self.gene_coexpr_mean + self.gene_coexpr_mean.transpose()
        self.gene_coexpr_var = self.gene_coexpr_var + self.gene_coexpr_var.transpose()

    def get_gene_delta_variance(self, gene_delta, return_abs=False):
        """
        :param gene_delta:
        :param return_abs:
        :return:
        """
        print('Getting gene delta variance over shortest paths...')
        try:
            self.hops
        except AttributeError:
            self.get_distance_matrix()

        n_parcels = self.data.shape[0]

        self.gene_delta_var = np.zeros((n_parcels, n_parcels))

        for i in np.arange(n_parcels):
            for j in np.arange(n_parcels):
                if j > i:
                    shortest_path = retrieve_shortest_path(i, j, self.hops, self.Pmat)
                    if len(shortest_path) != 0:
                        gene_delta_diff = np.diff(gene_delta[shortest_path[:, 0]], axis=0)

                        if return_abs == True:
                            gene_delta_diff = np.abs(gene_delta_diff)

                        # get the variance of the differences along the shortest path
                        self.gene_delta_var[i, j] = np.var(gene_delta_diff)
                    else:
                        self.gene_delta_var[i, j] = np.nan

        self.gene_delta_var = self.gene_delta_var + self.gene_delta_var.transpose()

class DataVector():
    def __init__(self, data=[]):
        self.data = data


    def regress_nuisance(self, c):
        x = self.data.reshape(-1, 1)
        c = c.reshape(-1, 1)

        nuis_reg = LinearRegression()
        nuis_reg.fit(c, x)

        x_pred = nuis_reg.predict(c)
        x_out = x[:, 0] - x_pred[:, 0]

        self.data_resid = x_out


def compute_fc(ts):
    """
    Parameters
    ----------
    ts : np.array (n_timepoints,n_parcels)
        time series

    Returns
    -------
    fc : np.array (n_parcels,n_parcels)
        functional connectivity matrix
    """

    fc = np.corrcoef(ts, rowvar=False)
    np.fill_diagonal(fc, np.nan)
    fc = np.arctanh(fc)
    np.fill_diagonal(fc, 1)

    return fc

def bandpower(ts, fs, fmin, fmax):
    """
    Helper function for compute_rlfp.

    Parameters
    ----------
    ts : np.array (n_timepoints,)
        time series
    fs : np.float
        sampling frequency
    fs : np.fmin
        minimum frequency of interest
    fs : np.fmax
        maximum frequency of interest
    Returns
    -------
    rlfp : np.float
        relative low frequency power
    """

    f, Pxx = sp.signal.periodogram(ts, fs=fs)
    ind_min = np.argmax(f > fmin) - 1
    ind_max = np.argmax(f > fmax) - 1

    return np.trapz(Pxx[ind_min: ind_max], f[ind_min: ind_max])

def compute_rlfp(ts, tr, num_bands=5, band_of_interest=1):
    """
    Parameters
    ----------
    ts : np.array (n_timepoints,)
        time series

    Returns
    -------
    rlfp : np.float
        relative low frequency power
    """

    num_timepoints = len(ts)

    scan_duration = num_timepoints * tr
    sample_freq = 1 / tr

    y = sp.stats.zscore(ts)

    band_intervals = np.linspace(0, sample_freq / 2, num_bands + 1)
    band_freq_range = band_intervals[band_of_interest - 1:band_of_interest + 1]

    return bandpower(y, sample_freq, band_freq_range[0], band_freq_range[1])
