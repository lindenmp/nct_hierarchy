# %% import
import sys, os, platform
from src.energy import simulate_natural_dynamics

import scipy as sp

# %% import workspace
os.environ["MY_PYTHON_WORKSPACE"] = 'ave_adj'
os.environ["WHICH_BRAIN_MAP"] = 'hist-g2'
from setup_workspace import *

# %% plotting
import seaborn as sns
import matplotlib.pyplot as plt
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
ax.set_ylabel('corr(gradient, activity)')
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
ax.set_ylabel('correlation differences')
ax.tick_params(pad=-2.5)
ax.grid(False)
sns.despine(right=True, top=True, ax=ax)
ax.set_xticks([])

f.savefig(os.path.join(environment.figdir, 'activity_propagation_hierarchy_corr_diff'),
          dpi=600, bbox_inches='tight', pad_inches=0.01)
plt.close()
