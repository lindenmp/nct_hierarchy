# %% import
import sys, os, platform
from src.plotting import my_distpair_plot
from src.energy import simulate_natural_dynamics
from src.utils import get_bootstrap_indices, mean_confidence_interval

import scipy as sp
from tqdm import tqdm

# %% import workspace
os.environ["MY_PYTHON_WORKSPACE"] = 'ave_adj'
os.environ["WHICH_BRAIN_MAP"] = 'hist-g2'
# os.environ["WHICH_BRAIN_MAP"] = 'func-g1'
from setup_workspace import *

# %% plotting
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
from src.plotting import set_plotting_params
set_plotting_params(format='svg')
figsize = 1.5

cmap = plt.cm.get_cmap('viridis', n_states)

# %% plots

# %% Panel B
ds = 0.5
activity = simulate_natural_dynamics(A, states, t0=0.01, h=50, ds=ds)

# correlation with gradient at each time point
f, ax = plt.subplots(1, 1, figsize=(figsize*2.5, figsize*1.2))
activity_corr = np.zeros((n_states, activity.shape[2]))
for i in np.arange(n_states):
    for j in np.arange(activity.shape[2]):
        activity_corr[i, j] = sp.stats.spearmanr(state_brain_map[states != i], activity[i, states != i, j])[0]
for i in np.arange(n_states):
    ax.plot(np.arange(1, activity.shape[2] + 1), activity_corr[i, :], color=cmap(i))

# axis options
ax.set_xlabel('t (a.u.)')
ax.set_ylabel('corr(hierarchy, dynamics)')
ax.tick_params(pad=-2.5)
ax.grid(False)
sns.despine(right=True, top=True, ax=ax)
ax.set_xticks([])

f.savefig(os.path.join(environment.figdir, 'activity_propagation_hierarchy_corr'),
          dpi=600, bbox_inches='tight', pad_inches=0.01)
plt.close()

# %% Panel C
# diff correlation with gradient at each time point
f, ax = plt.subplots(1, 1, figsize=(figsize*2.5, figsize*1.2))
for i in np.arange(n_states):
    ax.plot(np.arange(1, activity.shape[2]), np.diff(activity_corr[i, :]), color=cmap(i))

# axis options
ax.set_xlabel('t (a.u.)')
ax.set_ylabel('corr diff')
ax.tick_params(pad=-2.5)
ax.grid(False)
sns.despine(right=True, top=True, ax=ax)
ax.set_xticks([])

f.savefig(os.path.join(environment.figdir, 'activity_propagation_hierarchy_corr_diff'),
          dpi=600, bbox_inches='tight', pad_inches=0.01)
plt.close()

# %% Panel D
ds = 0.01
activity = simulate_natural_dynamics(A, states, t0=0.5, h=3.5, ds=ds)

activity_mean = np.zeros((n_states, n_states, activity.shape[2]))
for i in np.arange(n_states):
    activity_mean[:, i, :] = np.mean(activity[:, states == i, :], axis=1)

for s in [0, n_states - 1]:
    f, ax = plt.subplots(1, 1, figsize=(figsize*1.25, figsize*0.75))

    plot_data = activity_mean[s, :, :]
    peaks_t = np.argmax(plot_data, axis=1)
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

    f.tight_layout(pad=.75)
    f.savefig(os.path.join(environment.figdir, 'activity_from_x0-{0}_heatmap'.format(s)), dpi=300, bbox_inches='tight',
              pad_inches=0.01)
    plt.close()

# %% Panel E
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
    peak_gap[s] = np.sum(y)

f, ax = plt.subplots(1, 1, figsize=(figsize*1.15, figsize*1))
ax.scatter(x=np.arange(n_states), y=peak_corrs, c='gray', s=10, alpha=0.5)
print(sp.stats.spearmanr(np.arange(n_states),peak_corrs))
# axis options
ax.set_xlabel('', labelpad=-0.5)
ax.set_ylabel('', labelpad=-0.5)
ax.tick_params(pad=-2.5)
ax.grid(False)
sns.despine(right=True, top=True, ax=ax)
ax.set_xticks([])

f.tight_layout(pad=.75)
f.savefig(os.path.join(environment.figdir, 'activity_propagation'), dpi=300, bbox_inches='tight',
          pad_inches=0.01)
plt.close()

# %% Panel F
# run bootstrap
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
    ds = 0.1
    activity_strp = simulate_natural_dynamics(A_strp, states, t0=0.5, h=3.5, ds=ds)

    activity_mean_strp = np.zeros((n_states, n_states, activity_strp.shape[2]))
    for j in np.arange(n_states):
        activity_mean_strp[:, j, :] = np.mean(activity_strp[:, states == j, :], axis=1)

    for j, s in enumerate([0, n_states - 1]):
        a = activity_mean_strp[s, :, :]
        peaks_t = np.argmax(a, axis=1)
        peaks_t = np.delete(peaks_t, s)
        peak_corrs_bs[i, j], _ = sp.stats.spearmanr(np.arange(n_states - 1), peaks_t)

# plot
f, ax = plt.subplots(1, 1, figsize=(figsize*0.75, figsize*1.4))
df_plot = pd.DataFrame(data=np.abs(peak_corrs_bs),
                       columns=['bottom-up', 'top-down'])
my_distpair_plot(df_plot, ylabel='', ax=ax, test_stat=None, split=True)
ax.tick_params(pad=-2.5)
ax.set_ylabel('|$\\rho$|')
f.savefig(os.path.join(environment.figdir, 'activity_propagation_bootstrap'), dpi=300, bbox_inches='tight',
           pad_inches=0.01)
plt.close()

print(np.round(mean_confidence_interval(data=np.abs(peak_corrs_bs[:, 0])), 4))
print(np.round(mean_confidence_interval(data=np.abs(peak_corrs_bs[:, 1])), 4))
