import numpy as np
import scipy as sp
from scipy import stats
from scipy import signal
from bct.algorithms.distance import distance_wei_floyd
from sklearn.linear_model import LinearRegression

class DataMatrix():
    def __init__(self, data=[]):
        self.data = data

    def compute_distance_matrix(self):
        self.D, self.hops, self.Pmat = distance_wei_floyd(self.data, transform='inv')

    def regress_nuisance(self, c, indices=[]):
        if len(indices) == 0:
            # if no indices are given, assume filtering out of identity
            indices = np.where(~np.eye(self.data.shape[0], dtype=bool))

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
