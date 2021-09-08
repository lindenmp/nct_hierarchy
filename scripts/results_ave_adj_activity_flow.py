# %% import
import sys, os, platform
from pfactor_gradients.imaging_derivs import DataMatrix
from pfactor_gradients.pipelines import ComputeMinimumControlEnergy
from pfactor_gradients.plotting import my_reg_plot, my_distpair_plot, my_null_plot
from pfactor_gradients.energy import matrix_normalization, expand_states
from pfactor_gradients.utils import get_bootstrap_indices, mean_confidence_interval

import numpy as np
import pandas as pd
import scipy as sp
from scipy.linalg import svd
from tqdm import tqdm

# %% import workspace
from setup_workspace_ave_adj import *

# %% plotting
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
from pfactor_gradients.plotting import set_plotting_params
set_plotting_params(format='svg')
figsize = 1.5

# %% orthogonalize brain maps against state map
for key in load_average_bms.brain_maps:
    # load_average_bms.brain_maps[key].regress_nuisance(state_brain_map)
    # load_average_bms.brain_maps[key].data = load_average_bms.brain_maps[key].data_resid.copy()
    # load_average_bms.brain_maps[key].rankdata()
    # load_average_bms.brain_maps[key].rescale_unit_interval()
    print(key, sp.stats.pearsonr(state_brain_map, load_average_bms.brain_maps[key].data))

# bm = load_average_bms.brain_maps['ct'].data
# bm = load_average_bms.brain_maps['func-g1'].data
bm = state_brain_map

# %% propagation function
def simulate_natural_dynamics(A, states, h=5, ds=0.01):

    # Matrix normalization
    A_norm = matrix_normalization(A, version='continuous')

    x0_mat, xf_mat = expand_states(states)
    n_states = len(np.unique(states))
    x0s = xf_mat[:, :n_states]
    # bystanders = ~x0s

    ts = np.arange(0.5, h, ds)
    activity = np.zeros((n_states, n_parcels, len(ts)))
    # r = np.zeros((len(ts), n_states))

    for i, t in enumerate(ts):
        # scale normalized adj matrix to t
        A_s = A_norm * t

        # matrix exponential
        A_e = sp.linalg.expm(A_s)

        # get xfs
        xfs = np.matmul(A_e, x0s)
        activity[:, :, i] = xfs.transpose()

        # for j in np.arange(n_states):
        #     X = bm[bystanders[:, j]]
        #     y = xfs[bystanders[:, j], j]
        #     r[i, j], _ = sp.stats.spearmanr(X, y)

    activity_mean = np.zeros((n_states, n_states, len(ts)))
    for i in np.arange(n_states):
        # mean over target state regions
        x = np.mean(activity[:, states == i, :], axis=1)
        # store
        activity_mean[:, i, :] = x

    return activity, activity_mean

# %%
activity, activity_mean = simulate_natural_dynamics(A, states, h=3.5, ds=0.01)
print(activity.shape)

# %% bootstrap
n_samples = 500
bootstrap_indices = get_bootstrap_indices(d_size=n_subs, n_samples=n_samples)

peak_corrs_bs = np.zeros((n_samples, 2))

# loop start
for i in tqdm(np.arange(n_samples)):
    file_prefix = 'average_adj_n-{0}_cthr-{1}_smap-{2}_strap-{3}_'.format(load_average_sc.load_sc.df.shape[0],
                                                                consist_thresh, which_brain_map, i)

    load_sc_strap = LoadSC(environment=environment, Subject=Subject)
    load_sc_strap.df = load_sc.df.iloc[bootstrap_indices[i, :], :]
    load_sc_strap.A = load_sc.A[:, :, bootstrap_indices[i, :]]

    load_average_sc_strap = LoadAverageSC(load_sc=load_sc_strap, consist_thresh=consist_thresh, verbose=False)
    load_average_sc_strap.run()

    A_strp = load_average_sc_strap.A

    # simulate dynamics
    activity_strp, activity_mean_strp = simulate_natural_dynamics(A_strp, states, h=3.5, ds=0.1)

    for j, s in enumerate([0, 19]):
        a = activity_mean_strp[s, :, :]
        peaks_t = np.argmax(a, axis=1)
        peaks_t = np.delete(peaks_t, s)
        peak_corrs_bs[i, j], _ = sp.stats.spearmanr(np.arange(n_states - 1), peaks_t)

# %% plots

# %% 1) single states
for s in [0, 19]:
    f, ax = plt.subplots(1, 1, figsize=(figsize*1.25, figsize*0.75))

    plot_data = activity_mean[s, :, :]
    peaks_t = np.argmax(plot_data, axis=1)
    # peaks_t = np.delete(peaks_t, s)
    peaks_t = peaks_t.astype(float)
    peaks_t[s] = np.nan

    denom = np.max(plot_data, axis=1)
    denom = np.repeat(denom.reshape(-1, 1), plot_data.shape[1], axis=1)
    plot_data = np.divide(plot_data, denom)
    sns.heatmap(plot_data, square=False, ax=ax, cmap='rocket', cbar_kws={"shrink": 0.80, 'label': 'activity\n(a.u.)'})
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_xlabel('t (a.u.)')
    ax.set_yticks([])
    ax.tick_params(pad=-2.5)

    y = np.arange(n_states)+0.5
    rho, _ = sp.stats.spearmanr(np.delete(y, s), np.delete(peaks_t, s))
    textstr = '$\\rho$ = {:.2f}'.format(rho)
    ax.set_title(textstr, size=8)

    ax.scatter(peaks_t, y, marker='.', s=1, color='k')
    sns.regplot(x=peaks_t, y=y, ax=ax, ci=None, scatter=False, color='gray')
    # m, b = np.polyfit(y, peaks_t, 1)
    # ax.plot(y, m * y + b)

    f.tight_layout(pad=.75)
    f.savefig(os.path.join(environment.figdir, 'activity_from_x0-{0}_heatmap'.format(s)), dpi=300, bbox_inches='tight',
              pad_inches=0.01)
    plt.close()

# %% 2) state correlations
peak_corrs = np.zeros(n_states)
change_next = np.zeros(n_states)
peak_gap = np.zeros(n_states)
for s in np.arange(n_states):
    a = activity_mean[s, :, :]
    peaks = np.max(a, axis=1)
    peaks_t = np.argmax(a, axis=1)

    peaks = np.delete(peaks, s)
    peaks_t = np.delete(peaks_t, s)

    peak_corrs[s], _ = sp.stats.spearmanr(np.arange(n_states-1), peaks_t)

    ab = np.zeros((a.shape[0], a.shape[1] + 1))
    ab[:, :-1] = a
    ab[:, -1] = np.nan
    x = ab[np.arange(n_states-1), peaks_t+1] - peaks
    x = np.divide(x, np.abs(peaks)) * 100
    x = np.nanmean(x)
    change_next[s] = x

    y = np.abs(np.diff(peaks_t))
    # y = np.diff(peaks_t)
    peak_gap[s] = np.sum(y)

f, ax = plt.subplots(1, 1, figsize=(figsize*1.15, figsize*1))
# my_reg_plot(np.arange(n_states), peak_corrs, '', '', ax, annotate=None, regr_line=False, kde=False)
ax.scatter(x=np.arange(n_states), y=peak_corrs, c='gray', s=10, alpha=0.5)
# axis options
ax.set_xlabel('', labelpad=-0.5)
ax.set_ylabel('', labelpad=-0.5)
ax.tick_params(pad=-2.5)
ax.grid(False)
sns.despine(right=True, top=True, ax=ax)
ax.set_xticks([])
# my_reg_plot(np.arange(n_states), change_next, 'x0', 'change(peak,peak+1)', ax[1], annotate='both')
# my_reg_plot(np.arange(n_states), peak_gap, 'x0', 'gap', ax[2], annotate='both')

f.tight_layout(pad=.75)
f.savefig(os.path.join(environment.figdir, 'activity_propagation'), dpi=300, bbox_inches='tight',
          pad_inches=0.01)
plt.close()

# %% 3) bootstrapped results
f, ax = plt.subplots(1, 1, figsize=(figsize*0.75, figsize*1.4))
df_plot = pd.DataFrame(data=np.abs(peak_corrs_bs),
                       columns=['bottom-up', 'top-down'])
my_distpair_plot(df_plot, ylabel='', ax=ax, test_stat=None, split=True)
ax.tick_params(pad=-2.5)
ax.set_ylabel('|$\\rho$|')
f.savefig(os.path.join(environment.figdir, 'activity_propagation_bootstrap'), dpi=300, bbox_inches='tight',
          pad_inches=0.01)
plt.close()
