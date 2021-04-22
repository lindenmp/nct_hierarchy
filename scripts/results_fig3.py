import os
import numpy as np
import scipy as sp
import pandas as pd
from pfactor_gradients.pnc import Environment, Subject
from pfactor_gradients.routines import LoadSC, LoadCT, LoadRLFP, LoadCBF, LoadREHO, LoadALFF,\
    LoadAverageBrainMaps, LoadAverageSC
from pfactor_gradients.imaging_derivs import DataMatrix, DataVector
from pfactor_gradients.pipelines import ComputeGradients
from pfactor_gradients.plotting import my_regplot
from pfactor_gradients.utils import get_pdist_clusters, get_fdr_p
from tqdm import tqdm
from scipy import stats

# %% Plotting
import matplotlib.pyplot as plt
import seaborn as sns
from pfactor_gradients.plotting import roi_to_vtx
import nibabel as nib
from nilearn import plotting
sns.set(style='white', context='paper', font_scale=1)
import matplotlib.font_manager as font_manager
fontpath = '/Users/lindenmp/Library/Fonts/PublicSans-Thin.ttf'
prop = font_manager.FontProperties(fname=fontpath)
plt.rcParams['font.family'] = prop.get_name()
plt.rcParams['svg.fonttype'] = 'none'

# %% Setup project environment
computer = 'macbook'
parc = 'schaefer'
n_parcels = 400
sc_edge_weight = 'streamlineCount'
environment = Environment(computer=computer, parc=parc, n_parcels=n_parcels, sc_edge_weight=sc_edge_weight)
environment.make_output_dirs()
environment.load_parc_data()

# %% get clustered gradients
filters = {'healthExcludev2': 0, 't1Exclude': 0,
           'b0ProtocolValidationStatus': 1, 'dti64ProtocolValidationStatus': 1, 'dti64Exclude': 0,
           'psychoactiveMedPsychv2': 0, 'restProtocolValidationStatus': 1, 'restExclude': 0}
environment.load_metadata(filters)
compute_gradients = ComputeGradients(environment=environment, Subject=Subject)
compute_gradients.run()

# %% Load sc data
load_sc = LoadSC(environment=environment, Subject=Subject)
load_sc.run()
# refilter environment due to LoadSC excluding on disconnected nodes
environment.df = load_sc.df.copy()

# %% get minimum control energy
n_subs = environment.df.shape[0]
n_states = len(np.unique(compute_gradients.grad_bins))
mask = ~np.eye(n_states, dtype=bool)
indices = np.where(mask)

df_energy = pd.DataFrame()

if parc == 'schaefer' and n_parcels == 400:
    sparse_thresh = 0.06
elif parc == 'schaefer' and n_parcels == 200:
    sparse_thresh = 0.12

my_list = ['ct', 'cbf', 'reho', 'alff', 'wb']
for i in my_list:
    file = 'average_adj_n-{0}_s-{1}_ns-{2}-0_c-minimum_fast_T-1_B-{3}_E.npy'.format(n_subs, sparse_thresh, n_states, i)
    E = np.load(os.path.join(environment.pipelinedir, 'minimum_control_energy', file))
    df_energy[i] = E[indices]

# %% load null energy

n_perms = 1000
E_null = np.zeros((E.shape[0], E.shape[1], n_perms))
for i in np.arange(n_perms):
    file = 'average_adj_n-{0}_s-{1}_ns-{2}-0_c-minimum_fast_T-1_B-runi-{3}_E.npy'.format(n_subs, sparse_thresh, n_states, i)
    E_null[:, :, i] = np.load(os.path.join(environment.pipelinedir, 'minimum_control_energy', file))
    # df_estats_null['mean'].iloc[i] = np.mean(E_null[:, :, i][indices])
    # E_null_median[i] = np.median(E_null[:, :, i][indices])
    # E_null_var[i] = np.var(E_null[:, :, i][indices])

# %% get energy statistics
stats_list = ['mean', 'median', 'var', 'skew']
df_estats = pd.DataFrame(columns=my_list, index=stats_list)
df_estats_null = pd.DataFrame(columns=stats_list, index=np.arange(n_perms))

df_estats.loc['mean', :] = df_energy.mean()
df_estats.loc['median', :] = df_energy.median()
df_estats.loc['var', :] = df_energy.var()
df_estats.loc['skew', :] = df_energy.skew()

for i in np.arange(n_perms):
    df_estats_null.loc[i, 'mean'] = np.mean(E_null[:, :, i][indices])
    df_estats_null.loc[i, 'median'] = np.median(E_null[:, :, i][indices])
    df_estats_null.loc[i, 'var'] = np.var(E_null[:, :, i][indices])
    df_estats_null.loc[i, 'skew'] = sp.stats.skew(E_null[:, :, i][indices])

# %% fig 3a: energy distributions
cmap = sns.color_palette("Set2", as_cmap=False)
f, ax = plt.subplots(1, 1, figsize=(5, 2.5))
sns.violinplot(data=df_energy, ax=ax, palette=cmap, scale='width')
f.savefig(os.path.join(environment.figdir, 'fig-3a_energy_dist.png'), dpi=150, bbox_inches='tight',
          pad_inches=0.1)
plt.close()

# %% fig 3b: energy stats nulls
f, ax = plt.subplots(1, len(stats_list), figsize=(len(stats_list) * 2, 1))
for i, stat in enumerate(stats_list):
    sns.kdeplot(x=df_estats_null[stat].astype(float), ax=ax[i], bw_adjust=.75, clip_on=False, color='gray', alpha=0.5, linewidth=2)
    for j, energy in enumerate(my_list[:-1]):
        ax[i].axvline(x=df_estats.loc[stat, energy], ymax=0.25, clip_on=False, color=cmap[j], linewidth=2)

    for spine in ax[i].spines.values():
        spine.set_visible(False)
    ax[i].set_ylabel('')
    ax[i].set_yticklabels([])
    ax[i].set_yticks([])

f.subplots_adjust(hspace=1)
f.savefig(os.path.join(environment.figdir, 'fig-3b_energy_estats_nulls.png'), dpi=150, bbox_inches='tight',
          pad_inches=0.1)
plt.close()

# # %%
# f, ax = plt.subplots(1, 1, figsize=(2.5, 2.5))
# sns.histplot(data=E_null_var, ax=ax)
# for i in my_list:
#     ax.axvline(df_energy[i].var(), color='r')
#     x = np.min([np.sum(df_energy[i].var() > E_null_var),
#                 np.sum(df_energy[i].var() < E_null_var)]) / n_perms
#     print(i, x)
# f.savefig(os.path.join(environment.figdir, 'energy_null_moment.png'), dpi=150, bbox_inches='tight',
#           pad_inches=0.1)
# plt.close()
#
# # for i in my_list:
# #     for j in my_list:
# #         if i != j:
# #             if sp.stats.ttest_rel(df[i], df[j])[1] < 0.05:
# #                 print('t test', i, j, np.round(sp.stats.ttest_rel(df[i], df[j]), 4))
# #             if sp.stats.ks_2samp(df[i], df[j])[1] < 0.05:
# #                 print('ks test', i, j, np.round(sp.stats.ks_2samp(df[i], df[j]), 4))
