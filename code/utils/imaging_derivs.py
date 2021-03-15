import numpy as np
import scipy as sp
from scipy import stats
from scipy import signal

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
