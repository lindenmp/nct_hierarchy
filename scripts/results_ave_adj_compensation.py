# we want to understand whether there is asymmetry in the amount of compensation the control gramian has to perform
# between bottom-up and top-down transitions

# %% import
import stat
import sys, os, platform
from pfactor_gradients.imaging_derivs import DataMatrix
from pfactor_gradients.pipelines import ComputeMinimumControlEnergy
from pfactor_gradients.plotting import my_reg_plot, my_distpair_plot, my_null_plot
from pfactor_gradients.energy import matrix_normalization, expand_states, minimum_energy_fast
from pfactor_gradients.utils import get_bootstrap_indices, mean_confidence_interval, rank_int

import numpy as np
import pandas as pd
import scipy as sp
from scipy.linalg import svd
from tqdm import tqdm

import math


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
from pfactor_gradients.plotting import set_plotting_params
set_plotting_params(format='png')
figsize = 1.5

# %% brain maps
A_tmp = DataMatrix(data=A)
A_tmp.get_strength()
print(sp.stats.spearmanr(A_tmp.S, state_brain_map))

# get hierarchy distance between states
states_distance = np.zeros((n_states, n_states))
for i in np.arange(n_states):
    for j in np.arange(n_states):
        states_distance[i, j] = state_brain_map[states == i].mean() - state_brain_map[states == j].mean()
states_distance = np.abs(states_distance)
# states_distance = DataMatrix(data=states_distance)

# %% propagation function
def get_compensation(A_e, states):

    comp = np.zeros((n_states, n_states))

    for i in tqdm(np.arange(n_states)):
        x0 = states == i
        x0 = x0.reshape(-1, 1)

        for j in np.arange(n_states):
            xf = states == j
            xf = xf.reshape(-1, 1)

            tmp = (xf - np.matmul(A_e, x0))
            comp[i, j] = np.matmul(tmp.T, tmp)[0][0]

    return comp


def get_taylor_term(A_norm, taylor=1):

    if taylor == 0:
        taylor_term = np.eye(A_norm.shape[0])
    elif taylor == 1:
        taylor_term = A_norm.copy()
    else:
        taylor_term = np.linalg.matrix_power(A_norm, taylor) / math.factorial(taylor)

    return taylor_term


def get_taylor_adj(A_norm, n_taylor=5):

    A_e = np.zeros(A_norm.shape)

    for i in np.arange(n_taylor+1):
        A_e += get_taylor_term(A_norm=A_norm, taylor=i)

    return A_e


# %% check taylor convergence
# Matrix normalization
A_norm = matrix_normalization(A, version='continuous', c=1)

# matrix exponential
A_e = sp.linalg.expm(A_norm)

n = 10
ssd = np.zeros(n)

for i in np.arange(n):
    A_e_t = get_taylor_adj(A_norm, n_taylor=i+1)
    # A_e_t = get_taylor_term(A_norm, taylor=i+1)
    ssd[i] = np.sum(np.square(A_e_t.flatten() - A_e.flatten()))

f, ax = plt.subplots(1, 1, figsize=(figsize * 1.2, figsize * 1.2))
ax.plot(np.arange(1, n+1), ssd)
ax.set_ylabel("sum of square diffs (A_e, A_e_t)", labelpad=-1)
ax.set_xlabel("n_taylor", labelpad=-1)
ax.tick_params(pad=-2.5)
f.savefig(os.path.join(environment.figdir, 'plot'), dpi=600, bbox_inches='tight', pad_inches=0.01)
plt.close()

# %%
from pfactor_gradients.energy import control_energy_helper
# Matrix normalization
A_norm = matrix_normalization(A, version='continuous', c=1)

f, ax = plt.subplots(2, 4, figsize=(figsize * 6, figsize * 2.4))

for i in np.arange(4):
    # matrix exponential
    # A_e = get_taylor_adj(A_norm=A_norm, n_taylor=i+1)
    A_e = get_taylor_term(A_norm=A_norm, taylor=i+1)

    comp = get_compensation(A_e, states)
    # my_mat = comp
    my_mat = rank_int(comp) # normalized matrix

    # E, n_err = control_energy_helper(A=A_e, states=states, B=np.eye(n_parcels), T=1, control='minimum_fast')
    # my_mat = rank_int(E) # normalized matrix

    my_mat_asym = my_mat - my_mat.transpose()

    # plots
    # top-down vs bottom-up
    df_plot = pd.DataFrame(data=np.vstack((my_mat[indices_upper], my_mat[indices_lower])).transpose(),
                           columns=['bottom-up', 'top-down'])
    if i == 0:
        my_distpair_plot(df=df_plot, ylabel='compensation (z-score)', ax=ax[0, i])
    else:
        my_distpair_plot(df=df_plot, ylabel='', ax=ax[0, i])
    ax[0, i].title.set_text('n_taylor={0}'.format(i+1))

    # plot distance asymm
    my_reg_plot(states_distance[indices_upper], my_mat_asym[indices_upper],
                'hierarchy distance', 'comp. asymmetry', ax[1, i], annotate='spearman')


f.savefig(os.path.join(environment.figdir, 'taylor'), dpi=600, bbox_inches='tight',
          pad_inches=0.01)

# %%
# Matrix normalization
A_norm = matrix_normalization(A, version='continuous', c=1)

# matrix exponential
A_e = sp.linalg.expm(A_norm)

comp = get_compensation(A_e, states)
comp = rank_int(comp) # normalized matrix
comp_asym = comp - comp.transpose()

# plots
# 1) energy matrix
plot_mask = np.eye(n_states)
plot_mask = plot_mask.astype(bool)

f, ax = plt.subplots(1, 1, figsize=(figsize * 1.2, figsize * 1.2))
sns.heatmap(comp, mask=plot_mask, center=0, vmin=np.floor(np.min(comp[~plot_mask])), vmax=np.ceil(np.max(comp)),
            square=True, cmap='coolwarm', ax=ax, cbar_kws={"shrink": 0.80})
ax.set_ylabel("initial states", labelpad=-1)
ax.set_xlabel("target states", labelpad=-1)
ax.set_yticklabels('')
ax.set_xticklabels('')
ax.tick_params(pad=-2.5)
f.savefig(os.path.join(environment.figdir, 'comp'), dpi=600, bbox_inches='tight', pad_inches=0.01)
plt.close()

# 2) energy asymmetry

# top-down vs bottom-up
f, ax = plt.subplots(1, 1, figsize=(figsize, figsize))
df_plot = pd.DataFrame(data=np.vstack((comp[indices_upper], comp[indices_lower])).transpose(),
                       columns=['bottom-up', 'top-down'])
my_distpair_plot(df=df_plot, ylabel='compensation (z-score)', ax=ax)
f.savefig(os.path.join(environment.figdir, 'comp_asym'), dpi=600, bbox_inches='tight',
          pad_inches=0.01)
plt.close()

# energy asymmetry matrix
plot_mask = np.zeros((n_states, n_states))
plot_mask[indices_lower] = 1
plot_mask[np.eye(n_states) == 1] = 1
plot_mask = plot_mask.astype(bool)

f, ax = plt.subplots(1, 1, figsize=(figsize * 1.2, figsize * 1.2))
sns.heatmap(comp_asym, mask=plot_mask, center=0, vmin=np.floor(np.min(comp_asym)), vmax=np.ceil(np.max(comp_asym)),
            square=True, cmap='coolwarm', ax=ax, cbar_kws={"shrink": 0.80})
ax.set_ylabel("initial states", labelpad=-1)
ax.set_xlabel("target states", labelpad=-1)
ax.set_yticklabels('')
ax.set_xticklabels('')
ax.tick_params(pad=-2.5)
f.savefig(os.path.join(environment.figdir, 'comp_asym_matrix'), dpi=600, bbox_inches='tight',
          pad_inches=0.01)
plt.close()

# 3) energy asymmetry distance corr



# plot distance asymm
f, ax = plt.subplots(1, 1, figsize=(figsize, figsize))
my_reg_plot(states_distance[indices_upper], comp_asym[indices_upper],
            'hierarchy distance', 'compensation asymmetry', ax, annotate='spearman')
f.savefig(os.path.join(environment.figdir, 'corr(distance,comp_asym)'), dpi=600, bbox_inches='tight',
          pad_inches=0.01)
plt.close()