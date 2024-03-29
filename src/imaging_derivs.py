import os
import numpy as np
import scipy as sp
from scipy import stats
from scipy import signal
from bct.algorithms.distance import distance_wei, distance_wei_floyd, retrieve_shortest_path
from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge

# %% Plotting
import matplotlib.pyplot as plt
import seaborn as sns
from src.plotting import roi_to_vtx
import nibabel as nib
from nilearn import plotting
sns.set(style='white', context='talk', font_scale=1)
import matplotlib.font_manager as font_manager
try:
    fontpath = 'PublicSans-Thin.ttf'
    prop = font_manager.FontProperties(fname=fontpath)
    plt.rcParams['font.family'] = prop.get_name()
    plt.rcParams['svg.fonttype'] = 'none'
except:
    pass

class DataMatrix():
    def __init__(self, data=[], name=''):
        self.data = data
        self.name = name

    def get_distance_matrix(self, version='dijkstra'):
        if version == 'dijkstra':
            self.D, self.hops = distance_wei(1/self.data)
        elif version == 'floyd':
            self.D, self.hops, self.Pmat = distance_wei_floyd(self.data, transform='inv')

    def get_strength(self):
        self.S = np.sum(self.data, axis=0)

    def regress_nuisance(self, c, mask=[]):
        if len(mask) == 0:
            # if no indices are given, filter out identity and nans
            mask = ~np.eye(self.data.shape[0], dtype=bool) * ~np.isnan(self.data)
        indices = np.where(mask)

        x_out = np.zeros(self.data.shape)
        x_out[~mask] = np.nan

        x = self.data[indices].reshape(-1, 1)
        x_mean = np.nanmean(x, axis=0)
        c = c[indices].reshape(-1, 1)

        nuis_reg = LinearRegression()
        nuis_reg.fit(c, x)

        x_pred = nuis_reg.predict(c)
        x_out[indices] = x[:, 0] - x_pred[:, 0]
        x_out = x_out + x_mean

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

    def get_gradient_slopes(self, gradient, x_map=[], method='linear'):
        """
        :param gradient:
        :param x:
        :param method:
        :return:
        """
        print('Getting gradient slope over shortest paths...')
        if len(x_map) > 0:
            print('\tUsing external region map on x')

        self.gradient = gradient

        try:
            self.hops
        except AttributeError:
            self.get_distance_matrix()

        n_parcels = self.data.shape[0]

        self.grad_slope = np.zeros((n_parcels, n_parcels))
        self.grad_r2 = np.zeros((n_parcels, n_parcels))
        self.grad_resid = np.zeros((n_parcels, n_parcels))
        self.grad_var = np.zeros((n_parcels, n_parcels))

        for i in np.arange(n_parcels):
            for j in np.arange(n_parcels):
                if j > i:
                    shortest_path = retrieve_shortest_path(i, j, self.hops, self.Pmat)
                    if len(shortest_path) > 2:
                        try:
                            x = x_map[shortest_path[:, 0]].reshape(-1, 1)
                        except:
                            x = np.arange(len(shortest_path)).reshape(-1, 1)

                        y = gradient[shortest_path[:, 0]].reshape(-1, 1)

                        if method == 'linear':
                            self.grad_slope[i, j] = sp.stats.pearsonr(x.flatten(), y.flatten())[0]
                            reg = LinearRegression()
                        elif method == 'nonlinear':
                            self.grad_slope[i, j] = sp.stats.spearmanr(x.flatten(), y.flatten())[0]
                            reg = KernelRidge(kernel='rbf')

                        reg.fit(x, y)
                        self.grad_r2[i, j] = reg.score(x, y)

                        y_pred = reg.predict(x)
                        resid = y - y_pred

                        mse = np.mean(resid ** 2, axis=0)
                        rmse = np.sqrt(mse)
                        self.grad_resid[i, j] = rmse[0]

                        gradient_diff = np.diff(y, axis=0)
                        self.grad_var[i, j] = np.var(gradient_diff, axis=0)
                    else:
                        self.grad_slope[i, j] = np.nan
                        self.grad_r2[i, j] = np.nan
                        self.grad_resid[i, j] = np.nan
                        self.grad_var[i, j] = np.nan

        self.grad_slope = self.grad_slope + (self.grad_slope.transpose() * -1)
        self.grad_r2 = self.grad_r2 + self.grad_r2.transpose()
        self.grad_resid = self.grad_resid + self.grad_resid.transpose()
        self.grad_var = self.grad_var + self.grad_var.transpose()
        print('')

class DataVector():
    def __init__(self, data=[], name=''):
        self.data = data
        self.name = name


    def regress_nuisance(self, c):
        x = self.data
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if c.ndim == 1:
            c = c.reshape(-1, 1)

        nuis_reg = LinearRegression()
        nuis_reg.fit(c, x)

        x_pred = nuis_reg.predict(c)
        x_out = x[:, 0] - x_pred[:, 0]

        self.data_resid = x_out


    def rankdata(self, descending=False):
        nan_mask = np.isnan(self.data)

        tmp = np.zeros(len(self.data))
        tmp[nan_mask] = np.nan

        if descending:
            tmp[~nan_mask] = (len(self.data[~nan_mask]) + 1) - sp.stats.rankdata(self.data[~nan_mask])
        else:
            tmp[~nan_mask] = sp.stats.rankdata(self.data[~nan_mask])

        self.data = tmp.copy()


    def rescale_unit_interval(self):
        nan_mask = np.isnan(self.data)
        self.data[~nan_mask] = (self.data[~nan_mask] - min(self.data[~nan_mask])) / (max(self.data[~nan_mask]) - min(self.data[~nan_mask]))


    def shuffle_data(self, n_shuffles=10000, shuffle_indices=[]):
        if len(shuffle_indices) == 0:
            # for reproducibility
            np.random.seed(0)
            data_shuf = np.zeros((len(self.data), n_shuffles))
            for i in np.arange(n_shuffles):
                idx = np.arange(0, len(self.data))
                np.random.shuffle(idx)
                data_shuf[:, i] = self.data[idx].copy()
        else:
            n_shuffles = shuffle_indices.shape[1]
            data_shuf = np.zeros((len(self.data), n_shuffles))
            for i in np.arange(n_shuffles):
                data_shuf[:, i] = self.data[shuffle_indices[:, i]].copy()

        self.data_shuf = data_shuf


    def mean_between_states(self, states):
        x = self.data

        unique = np.unique(states, return_counts=False)
        n_states = len(unique)

        x_out = np.zeros((n_states, n_states))
        for i in np.arange(n_states):
            for j in np.arange(n_states):
                x_out[i, j] = np.nanmean(x[np.logical_or(states == i, states == j)])

        self.data_mean = x_out


    def mean_within_states(self, states):
        x = self.data

        unique = np.unique(states, return_counts=False)
        n_states = len(unique)

        x_out = np.zeros(n_states)
        for i in np.arange(n_states):
            x_out[i] = np.nanmean(x[states == i])

        # tile to match target states (columns)
        x_out = np.tile(x_out, (n_states, 1))

        self.data_mean = x_out


    def brain_surface_plot(self, environment, cmap='viridis', order='lr'):
        f, ax = plt.subplots(1, 4, subplot_kw={'projection': '3d'})
        data = self.data.copy()

        if np.min(data) == 0:
            data = data + 1e-5

        n = int(data.shape[0] / 2)
        if order == 'lr':
            roi_data_lh = data[:n]
            roi_data_rh = data[n:]
        elif order == 'rl':
            roi_data_rh = data[:n]
            roi_data_lh = data[n:]

        vtx_data, plot_min, plot_max = roi_to_vtx(roi_data_lh, environment.lh_annot_file)
        vtx_data = vtx_data.astype(float)
        plotting.plot_surf_roi(environment.fsaverage['infl_left'], roi_map=vtx_data,
                               hemi='left', view='lateral', vmin=plot_min, vmax=plot_max,
                               bg_map=environment.fsaverage['sulc_left'], bg_on_data=True, axes=ax[0],
                               darkness=.5, cmap=cmap, colorbar=False)

        plotting.plot_surf_roi(environment.fsaverage['infl_left'], roi_map=vtx_data,
                               hemi='left', view='medial', vmin=plot_min, vmax=plot_max,
                               bg_map=environment.fsaverage['sulc_left'], bg_on_data=True, axes=ax[1],
                               darkness=.5, cmap=cmap, colorbar=False)

        vtx_data, plot_min, plot_max = roi_to_vtx(roi_data_rh, environment.rh_annot_file)
        vtx_data = vtx_data.astype(float)
        plotting.plot_surf_roi(environment.fsaverage['infl_right'], roi_map=vtx_data,
                               hemi='right', view='lateral', vmin=plot_min, vmax=plot_max,
                               bg_map=environment.fsaverage['sulc_right'], bg_on_data=True, axes=ax[2],
                               darkness=.5, cmap=cmap, colorbar=False)

        plotting.plot_surf_roi(environment.fsaverage['infl_right'], roi_map=vtx_data,
                               hemi='right', view='medial', vmin=plot_min, vmax=plot_max,
                               bg_map=environment.fsaverage['sulc_right'], bg_on_data=True, axes=ax[3],
                               darkness=.5, cmap=cmap, colorbar=True)
        # plt.show()
        plt.subplots_adjust(wspace=0, hspace=0)
        f.savefig(os.path.join(environment.figdir, self.name+'.png'), dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()

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

def compute_rlfp(ts, tr, low=None, high=None, num_bands=5, band_of_interest=1):
    """
    Parameters
    ----------
    ts : np.array (n_timepoints,)
        time series
    tr : np.float
        tr in seconds

    Returns
    -------
    rlfp : np.float
        relative low frequency power
    """

    sample_freq = 1 / tr
    nyq_freq = sample_freq / 2

    y = sp.stats.zscore(ts)

    if low is None and high is None:
        band_intervals = np.linspace(0, nyq_freq, num_bands + 1)
    else:
        band_intervals = np.linspace(low, high, num_bands + 1)

    band_freq_range = band_intervals[band_of_interest - 1:band_of_interest + 1]

    return bandpower(y, sample_freq, band_freq_range[0], band_freq_range[1])

def compute_transition_probs_updown(rsts_labels, states, n_steps=1):
    unique = np.unique(states)
    n_states = len(unique)
    n_trs = len(rsts_labels)

    probs_up = np.zeros(n_states)
    probs_down = np.zeros(n_states)

    for i in np.arange(n_states):
        try:
            state_idx = np.where(rsts_labels == i)[0]

            up = np.zeros(len(state_idx)).astype(bool)
            for j in np.arange(1, n_steps + 1):
                # tmp = rsts_labels[state_idx + 1] == i + j
                tmp = rsts_labels[state_idx + j] == i + 1
                up = up + tmp
            probs_up[i] = np.sum(up) / len(state_idx)

            down = np.zeros(len(state_idx)).astype(bool)
            for j in np.arange(1, n_steps + 1):
                # tmp = rsts_labels[state_idx + 1] == i - j
                tmp = rsts_labels[state_idx + j] == i - 1
                down = down + tmp
            probs_down[i] = np.sum(down) / len(state_idx)
        except:
            probs_up[i] = np.nan
            probs_down[i] = np.nan

    probs_ratio = probs_up / probs_down

    return probs_up, probs_down, probs_ratio


# def compute_transition_probs_updown(rsts_labels, states):
#     unique = np.unique(states)
#     n_states = len(unique)
#     n_TRs = len(rsts_labels)
#     probabilities = np.zeros(n_states)
#
#     for i in np.arange(n_states):
#         try:
#             state_idx = np.where(rsts_labels == i)[0]
#             if n_TRs - 1 in state_idx:
#                 state_idx = np.delete(state_idx, state_idx == n_TRs - 1)
#             probabilities[i] = np.sum(rsts_labels[state_idx + 1] >= rsts_labels[state_idx]) / len(state_idx)
#         except:
#             probabilities[i] = np.nan
#
#     return probabilities