import os
import numpy as np
import scipy as sp
import pandas as pd
from pfactor_gradients.pnc import Environment, Subject
from pfactor_gradients.routines import LoadSC, LoadCT, LoadRLFP
from pfactor_gradients.imaging_derivs import DataMatrix, DataVector
from pfactor_gradients.pipelines import ComputeGradients
from pfactor_gradients.plotting import my_regplot
from pfactor_gradients.utils import get_pdist_clusters, get_exact_p

# %% Plotting
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='white', context='talk', font_scale=1)
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

# %% load ct data
load_ct = LoadCT(environment=environment, Subject=Subject)
load_ct.run()

ct = DataVector(data=np.nanmean(load_ct.ct, axis=0), name='ct')
ct.rankdata()
ct.rescale_unit_interval()

# %% load rlfp data
load_rlfp = LoadRLFP(environment=environment, Subject=Subject)
load_rlfp.run()

rlfp = DataVector(data=np.nanmean(load_rlfp.rlfp, axis=0), name='rlfp')
rlfp.rankdata()
rlfp.rescale_unit_interval()

# %%  1) mni/gradient distance between hierarchy clusters
dist_mni = get_pdist_clusters(environment.centroids.values, compute_gradients.kmeans.labels_, method='median')
dist_mni[np.eye(dist_mni.shape[0]) == 1] = np.nan

dist_h = sp.spatial.distance.squareform(sp.spatial.distance.pdist(compute_gradients.kmeans.cluster_centers_))
dist_h[np.eye(dist_h.shape[0]) == 1] = np.nan

# Plot
f, ax = plt.subplots(1, 3, figsize=(15, 4))
sns.heatmap(dist_mni, ax=ax[0], square=True)
ax[0].set_title('Distance (MNI)')
sns.heatmap(dist_h, ax=ax[1], square=True)
ax[1].set_title('Distance (Hierarchy)')
my_regplot(dist_mni, dist_h, 'Distance (MNI)', 'Distance (Hierarchy)', ax[2])
f.savefig(os.path.join(environment.figdir, 'distance_vs_distance.png'), dpi=150, bbox_inches='tight', pad_inches=0.1)
plt.close()

# %% get minimum control energy
mask = ~np.eye(compute_gradients.kmeans.n_clusters, dtype=bool)
indices = np.where(mask)

files = ['average_adj_n-775_s-0.06_ns-20-20_c-minimum_fast_T-1_B-wb_E.npy',
         'average_adj_n-775_s-0.06_ns-20-20_c-minimum_fast_T-1_B-ct_E.npy',
         'average_adj_n-775_s-0.06_ns-20-20_c-minimum_fast_T-1_B-rlfp_E.npy',
         ]
# 'noise_average_adj_n-775_s-0.06_ns-20-20_c-minimum_fast_T-1_B-wb_E.npy',

column_labels = ['identity', 'ct', 'rlfp']

df_plot = pd.DataFrame()

for i, file in enumerate(files):
    E = np.load(os.path.join(environment.pipelinedir, 'minimum_control_energy', file))
    E = DataMatrix(data=E)
    df_plot[column_labels[i]] = E.data[indices]

# unique, counts = np.unique(compute_gradients.kmeans.labels_, return_counts=True)
# # np.tile(counts, (len(counts), 1))
# for i in np.arange(20):
#     mask = ~np.isnan(E.data[i, :])
#     print(sp.stats.pearsonr(counts[mask], E.data[i, :][mask]))

# %% Plots

# %% energy distributions
f, ax = plt.subplots(1, 1, figsize=(5, 5))
sns.violinplot(data=df_plot, ax=ax)
f.savefig(os.path.join(environment.figdir, 'energy_dist.png'), dpi=150, bbox_inches='tight',
          pad_inches=0.1)
plt.close()

print(df_plot['ct'].mean() - df_plot['rlfp'].mean(),
      get_exact_p(df_plot['ct'], df_plot['rlfp']),
      sp.stats.ttest_rel(df_plot['ct'], df_plot['rlfp']))

# %% Does energy vary over the state transitions according to the neurobiological feature that is used for optimization?
f, ax = plt.subplots(1, 1, figsize=(5, 5))
my_regplot(ct.data, rlfp.data, 'ct', 'rlfp', ax)
f.savefig(os.path.join(environment.figdir, 'corr(ct,rlfp).png'), dpi=150, bbox_inches='tight',
          pad_inches=0.1)
plt.close()

ct.mean_over_clusters(compute_gradients.kmeans.labels_)
rlfp.mean_over_clusters(compute_gradients.kmeans.labels_)

f, ax = plt.subplots(1, 1, figsize=(5, 5))
my_regplot(ct.data_clusters, rlfp.data_clusters, 'ct (states)', 'rlfp (states)', ax)
f.savefig(os.path.join(environment.figdir, 'corr(ct_states,rlfp_states).png'), dpi=150, bbox_inches='tight',
          pad_inches=0.1)
plt.close()

f, ax = plt.subplots(1, 1, figsize=(5, 5))
my_regplot(ct.data_clusters[indices], df_plot['ct'], 'ct', 'Energy (ct)', ax)
f.savefig(os.path.join(environment.figdir, 'corr(ct,energy_ct).png'), dpi=150, bbox_inches='tight',
          pad_inches=0.1)
plt.close()

f, ax = plt.subplots(1, 1, figsize=(5, 5))
my_regplot(rlfp.data_clusters[indices], df_plot['rlfp'], 'rlfp', 'Energy (rlfp)', ax)
f.savefig(os.path.join(environment.figdir, 'corr(rlfp,energy_rlfp).png'), dpi=150, bbox_inches='tight',
          pad_inches=0.1)
plt.close()

# %% Does energy correlate across different maps?
f, ax = plt.subplots(1, 1, figsize=(5, 5))
my_regplot(df_plot['identity'], df_plot['ct'], 'Energy (identity)', 'Energy (ct)', ax)
f.savefig(os.path.join(environment.figdir, 'corr(energy_identity,energy_ct).png'), dpi=150, bbox_inches='tight',
          pad_inches=0.1)
plt.close()

f, ax = plt.subplots(1, 1, figsize=(5, 5))
my_regplot(df_plot['identity'], df_plot['rlfp'], 'Energy (identity)', 'Energy (rlfp)', ax)
f.savefig(os.path.join(environment.figdir, 'corr(energy_identity,energy_rlfp).png'), dpi=150, bbox_inches='tight',
          pad_inches=0.1)
plt.close()

f, ax = plt.subplots(1, 1, figsize=(5, 5))
my_regplot(df_plot['ct'], df_plot['rlfp'], 'Energy (ct)', 'Energy (rlfp)', ax)
f.savefig(os.path.join(environment.figdir, 'corr(energy_ct,energy_rlfp).png'), dpi=150, bbox_inches='tight',
          pad_inches=0.1)
plt.close()
