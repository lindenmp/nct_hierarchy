# %% import
import sys, os, platform
from pfactor_gradients.imaging_derivs import DataMatrix
from pfactor_gradients.pipelines import ComputeMinimumControlEnergy
from pfactor_gradients.plotting import my_reg_plot, my_distpair_plot, my_null_plot
from pfactor_gradients.energy import expand_states
from pfactor_gradients.utils import rank_int, get_null_p

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

# %%
from sklearn.linear_model import LinearRegression

# %%
c = 1
# singluar value decomposition
u, s, vt = svd(A)
# Matrix normalization
A_norm = A / (c + s[0]) - np.eye(n_parcels)

x0_mat, xf_mat = expand_states(states)
x0s = xf_mat[:, :n_states]
bystanders = ~x0s

ts = np.arange(0.1, 40, 1)
r = np.zeros((len(ts), n_states))

for i, t in tqdm(enumerate(ts)):
    # scale normalized adj matrix to t
    A_s = A_norm * t

    # matrix exponential
    A_e = sp.linalg.expm(A_s)

    # get xfs
    xfs = np.matmul(A_e, x0s)

    for j in np.arange(n_states):
        X = bm[bystanders[:, j]]
        y = xfs[bystanders[:, j], j]
        # reg = LinearRegression().fit(X.reshape(-1, 1), y.reshape(-1, 1))
        # r[i, j] = reg.score(X.reshape(-1, 1), y.reshape(-1, 1))
        r[i, j], _ = sp.stats.spearmanr(X, y)

# %% plot
# cmap = plt.cm.get_cmap('Blues', n_states)
cmap = plt.cm.get_cmap('viridis', n_states)

f, ax = plt.subplots(1, 1, figsize=(figsize, figsize))
for i in np.arange(n_states):
    ax.plot(ts, r[:, i], color=cmap(i))
ax.set_xlabel('t')
ax.set_ylabel('corr(activity,hierarchy)')
ax.tick_params(pad=-2.5)
f.savefig(os.path.join(environment.figdir, 'activity_from_x0'), dpi=600, bbox_inches='tight',
          pad_inches=0.01)
plt.close()

# %%
f, ax = plt.subplots(1, 1, figsize=(figsize, figsize))
my_reg_plot(np.arange(n_states), r[0, :], 'states', 'corr(activity,hierarchy)', ax, annotate='both')
ax.set_title('t = 0.1')
ax.tick_params(pad=-2.5)
f.savefig(os.path.join(environment.figdir, 'activity_from_x0_t-01'), dpi=600, bbox_inches='tight',
          pad_inches=0.01)
plt.close()

# # %%
# c = 1
# # singluar value decomposition
# u, s, vt = svd(A)
# A_norm = A / (c + s[0]) - np.eye(n_parcels)
#
# w, v = sp.linalg.eig(A_norm)
# w = w.astype(float)
#
# r_eig = np.zeros(n_parcels)
# for i in np.arange(n_parcels):
#     r_eig[i], _ = sp.stats.spearmanr(bm, v[:, i])
#
# f, ax = plt.subplots(1, 1, figsize=(figsize, figsize))
# my_reg_plot(w, np.abs(r_eig), 'eig vals', 'corr(eig,hierarchy)', ax, annotate='both')
#
# # sns.histplot(x=r_eig, ax=ax, color='gray')
# # ax.grid(False)
# # sns.despine(right=True, top=True, ax=ax)
# # ax.tick_params(pad=-2.5)
# # ax.set_xlabel('corr(eig,hierarchy)', labelpad=-0.5)
# # ax.set_ylabel('counts', labelpad=-0.5)
# f.savefig(os.path.join(environment.figdir, 'corr(eig,hierarchy)'), dpi=600, bbox_inches='tight',
#           pad_inches=0.01)
# plt.close()
#
# # %%
# # idx_sort = sp.stats.rankdata(state_brain_map).astype(int) - 1
# idx_sort = sp.stats.rankdata(bm).astype(int) - 1
# f, ax = plt.subplots(1, 3, figsize=(figsize*3, figsize))
# h2 = []
# for i in np.arange(1, 4):
#     idx = np.argsort(w)[-i]
#     # h = np.matmul(v[:, idx].reshape(-1, 1), v[:, idx].transpose().reshape(1, -1))
#     h = v[:, i].reshape(-1, 1) * v[:, i].transpose().reshape(1, -1)
#     h2 =+ h
#     h = rank_int(h)
#     h = h[idx_sort, :][:, idx_sort]
#     sns.heatmap(h, ax=ax[i-1], square=True, center=0)
#     ax[i-1].set_yticklabels('')
#     ax[i-1].set_xticklabels('')
#     ax[i-1].set_title('eig value {:.3f}'.format(w[idx]))
# f.savefig(os.path.join(environment.figdir, 'heatmap'), dpi=600, bbox_inches='tight',
#           pad_inches=0.01)
# plt.close()
#
# f, ax = plt.subplots(1, 1, figsize=(figsize, figsize))
# h2 = rank_int(h2)
# h2 = h2[idx_sort, :][:, idx_sort]
# sns.heatmap(h2, ax=ax, square=True, center=0)
# ax.set_yticklabels('')
# ax.set_xticklabels('')
# f.savefig(os.path.join(environment.figdir, 'heatmap2'), dpi=600, bbox_inches='tight',
#           pad_inches=0.01)
# plt.close()
