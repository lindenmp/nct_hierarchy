# %% import
import sys, os, platform
from pfactor_gradients.imaging_derivs import DataMatrix
from pfactor_gradients.pipelines import ComputeMinimumControlEnergy
from pfactor_gradients.plotting import my_reg_plot, my_distpair_plot
from pfactor_gradients.energy import expand_states

import numpy as np
import pandas as pd
import scipy as sp
from scipy.linalg import svd

# %% import workspace
from setup_workspace_ave_adj import *

# %% plotting
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
from pfactor_gradients.plotting import set_plotting_params
set_plotting_params(format='png')
figsize = 1.5

# %% orthogonalize brain maps against state map
for key in load_average_bms.brain_maps:
    # load_average_bms.brain_maps[key].regress_nuisance(state_brain_map)
    # load_average_bms.brain_maps[key].data = load_average_bms.brain_maps[key].data_resid.copy()
    # load_average_bms.brain_maps[key].rankdata()
    # load_average_bms.brain_maps[key].rescale_unit_interval()
    print(key, sp.stats.pearsonr(state_brain_map, load_average_bms.brain_maps[key].data))

# %% get control energy
file_prefix = 'average_adj_n-{0}_s-{1}_{2}_'.format(load_average_sc.load_sc.df.shape[0], spars_thresh, which_brain_map)
n_subsamples = 0
# B_list = ['wb',] + list(load_average_bms.brain_maps.keys())
B_list = ['wb',]
E = dict.fromkeys(B_list)

for B in B_list:
    if B == 'wb':
        nct_pipeline = ComputeMinimumControlEnergy(environment=environment, A=A,
                                                   states=states, n_subsamples=n_subsamples,
                                                   control='minimum_fast', T=1, B='wb', file_prefix=file_prefix,
                                                   force_rerun=True, save_outputs=False, verbose=True)
        nct_pipeline.run()
    else:
        nct_pipeline = ComputeMinimumControlEnergy(environment=environment, A=A,
                                                   states=states, n_subsamples=n_subsamples,
                                                   control='minimum_fast', T=1, B=load_average_bms.brain_maps[B],
                                                   file_prefix=file_prefix,
                                                   force_rerun=True, save_outputs=False, verbose=True)
        nct_pipeline.run()

    E[B] = nct_pipeline.E

# %% plots

# %% data for plotting
# for B in ['wb', 'ct', 'sa', 'func-g1']:
B = 'wb'
print(B)
# e = rank_int(E[B]) # normalized energy matrix
e = E[B] # normalized energy matrix
ed = e.transpose() - e # energy asymmetry matrix
# save out mean ed for use in other scripts
np.save(os.path.join(environment.pipelinedir, 'e_{0}_{1}.npy'.format(which_brain_map, B)), e)
np.save(os.path.join(environment.pipelinedir, 'ed_{0}_{1}.npy'.format(which_brain_map, B)), ed)
# np.save(os.path.join(environment.pipelinedir, 'e_{0}_{1}_gi.npy'.format(which_brain_map, B)), e)
# np.save(os.path.join(environment.pipelinedir, 'ed_{0}_{1}_gi.npy'.format(which_brain_map, B)), ed)

# %% 1) energy dists: top-down vs bottom-up
f, ax = plt.subplots(1, 1, figsize=(figsize, figsize))
df_plot = pd.DataFrame(data=np.vstack((e[indices_upper], e[indices_lower])).transpose(),
                       columns=['bottom-up', 'top-down'])
my_distpair_plot(df=df_plot, ylabel='energy (z-score)', ax=ax)
f.savefig(os.path.join(environment.figdir, 'e_{0}'.format(B)), dpi=600, bbox_inches='tight',
          pad_inches=0.01)
plt.close()

# 1.1) energy matrix
plot_mask = np.eye(n_states)
plot_mask = plot_mask.astype(bool)

f, ax = plt.subplots(1, 1, figsize=(figsize*1.2, figsize*1.2))
sns.heatmap(e, mask=plot_mask, vmin=np.floor(np.min(e[~plot_mask])), vmax=np.ceil(np.max(e)),
                        square=True, ax=ax, cbar_kws={"shrink": 0.80})
ax.set_ylabel("initial states", labelpad=-1)
ax.set_xlabel("target states", labelpad=-1)
ax.set_yticklabels('')
ax.set_xticklabels('')
ax.tick_params(pad=-2.5)
f.savefig(os.path.join(environment.figdir, 'e_{0}'.format(B)), dpi=600, bbox_inches='tight', pad_inches=0.01)
plt.close()

# 2) energy asymmetry matrix
plot_mask = np.zeros((n_states, n_states))
plot_mask[indices_upper] = 1
plot_mask[np.eye(n_states) == 1] = 1
plot_mask = plot_mask.astype(bool)

f, ax = plt.subplots(1, 1, figsize=(figsize*1.2, figsize*1.2))
# sns.heatmap(rank_int(ed), mask=plot_mask, center=0, vmin=-2, vmax=2,
sns.heatmap(ed, mask=plot_mask, center=0, vmin=np.floor(np.min(ed)), vmax=np.ceil(np.max(ed)),
                        square=True, cmap='coolwarm', ax=ax, cbar_kws={"shrink": 0.80})
ax.set_ylabel("initial states", labelpad=-1)
ax.set_xlabel("target states", labelpad=-1)
ax.set_yticklabels('')
ax.set_xticklabels('')
ax.tick_params(pad=-2.5)
f.savefig(os.path.join(environment.figdir, 'e_asym_{0}'.format(B)), dpi=600, bbox_inches='tight', pad_inches=0.01)
plt.close()

# %% 2.1) energy asymmetry distance

# get hierarchy distance between states
states_distance = np.zeros((n_states, n_states))
for i in np.arange(n_states):
    for j in np.arange(n_states):
        states_distance[i, j] = state_brain_map[states == i].mean() - state_brain_map[states == j].mean()
states_distance = DataMatrix(data=states_distance)

# get mni distance between states
states_distance_mni = sp.spatial.distance.squareform(sp.spatial.distance.pdist(environment.centroids.values))
states_distance_mni[np.eye(states_distance_mni.shape[0]) == 1] = np.nan
states_distance_mni = DataMatrix(data=states_distance_mni)
states_distance_mni.mean_over_clusters(states)

# regress mni distance out of energy asymmetry
ed_matrix = DataMatrix(data=ed)
mask = np.zeros((n_states, n_states)).astype(bool)
mask[indices_lower] = True
ed_matrix.regress_nuisance(c=states_distance_mni.data_clusters, mask=mask)

# plot
f, ax = plt.subplots(1, 1, figsize=(figsize, figsize))
my_reg_plot(states_distance.data[indices_lower], np.abs(ed_matrix.data_resid[indices_lower]),
                        'hierarchy distance', 'energy asymmetry\n(abs.)', ax, annotate='spearman')
f.savefig(os.path.join(environment.figdir, 'corr(distance,abs_e_asym_{0})'.format(B)), dpi=600, bbox_inches='tight',
          pad_inches=0.01)
plt.close()

# plot
f, ax = plt.subplots(1, 1, figsize=(figsize, figsize))
my_reg_plot(states_distance.data[indices_lower], ed_matrix.data_resid[indices_lower],
            'hierarchy distance', 'energy asymmetry', ax, annotate='spearman')
f.savefig(os.path.join(environment.figdir, 'corr(distance,e_asym_{0})'.format(B)), dpi=600, bbox_inches='tight',
          pad_inches=0.01)
plt.close()

# %%
c = 1
# singluar value decomposition
u, s, vt = svd(A)
# Matrix normalization
A_norm = A / (c + s[0]) - np.eye(n_parcels)

x0_mat, xf_mat = expand_states(states)
xf_mat_tmp = xf_mat[:, :n_states]
xf_mat_tmp = xf_mat_tmp.astype(int)

A_e = sp.linalg.expm(A_norm) # matrix exponential

X = np.matmul(A_e, xf_mat_tmp)
X2 = np.sum(np.square(X), axis=0)

# plot
f, ax = plt.subplots(1, 4, figsize=(figsize*4, figsize))
ax[0].plot(np.arange(n_parcels), X[:, 10])
my_reg_plot(np.arange(n_states), X2, 'states', 'X', ax[1], annotate='both')
my_reg_plot(np.sum(e, axis=1), X2, 'energy (initial)', 'X', ax[2], annotate='both')
my_reg_plot(np.sum(e, axis=0), X2, 'energy (target)', 'X', ax[3], annotate='both')
f.savefig(os.path.join(environment.figdir, 'test'), dpi=600, bbox_inches='tight',
          pad_inches=0.01)
plt.close()

# %%
dm = DataMatrix(data=A)
dm.get_strength()
dm.get_distance_matrix()
from pfactor_gradients.utils import mean_over_clusters
hops = mean_over_clusters(dm.hops, states)
D = mean_over_clusters(dm.D, states)

# get hierarchy distance between states
S_states = np.zeros(n_states)
for i in np.arange(n_states):
    S_states[i] = np.mean(dm.S[states == i])

f, ax = plt.subplots(1, 4, figsize=(figsize*5, figsize))
my_reg_plot(x=state_brain_map, y=dm.S,
            xlabel='hierarchy position', ylabel='strength (regional)', ax=ax[0], annotate='both')
my_reg_plot(x=np.arange(n_states), y=S_states,
            xlabel='states', ylabel='strength (state ave.)', ax=ax[1], annotate='both')
my_reg_plot(np.sum(e, axis=1), S_states, 'energy (initial)', 'strength (state ave.)', ax[2], annotate='both')
my_reg_plot(np.sum(e, axis=0), S_states, 'energy (target)', 'strength (state ave.)', ax[3], annotate='both')
f.savefig(os.path.join(environment.figdir, 'corr(states,strength)'), dpi=600, bbox_inches='tight',
          pad_inches=0.01)
plt.close()
