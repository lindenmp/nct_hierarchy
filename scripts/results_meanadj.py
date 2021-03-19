import os

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import seaborn as sns
from sklearn.cluster import KMeans

from data_loader.routines import LoadSC, LoadRLFP
from utils.imaging_derivs import DataMatrix, DataVector
from utils.plotting import my_regplot, my_nullplot
from utils.utils import get_pdist_clusters, get_disc_repl, mean_over_clusters

# %% Set general plotting params
sns.set(style='white', context='talk', font_scale=1)
import matplotlib.font_manager as font_manager

fontpath = '/Users/lindenmp/Library/Fonts/PublicSans-Thin.ttf'
prop = font_manager.FontProperties(fname=fontpath)
plt.rcParams['font.family'] = prop.get_name()
plt.rcParams['svg.fonttype'] = 'none'

# %% Setup project environment
from data_loader.pnc import Environment, Subject

parc = 'schaefer'
n_parcels = 400
sc_edge_weight = 'streamlineCount'
environment = Environment(parc=parc, n_parcels=n_parcels, sc_edge_weight=sc_edge_weight)
environment.make_output_dirs()
environment.load_parc_data()

# %% get clustered gradients
from data_loader.pipelines import ComputeGradients

filters = {'healthExcludev2': 0, 't1Exclude': 0,
           'b0ProtocolValidationStatus': 1, 'dti64ProtocolValidationStatus': 1, 'dti64Exclude': 0,
           'psychoactiveMedPsychv2': 0, 'restProtocolValidationStatus': 1, 'restExclude': 0}
environment.load_metadata(filters)
cg = ComputeGradients(environment=environment, Subject=Subject)
cg.run()

# %%  1) mni/gradient distance between hierarchy clusters
dist_mni = get_pdist_clusters(environment.centroids.values, cg.kmeans.labels_, method='median')
dist_mni[np.eye(dist_mni.shape[0]) == 1] = np.nan

dist_h = sp.spatial.distance.squareform(sp.spatial.distance.pdist(cg.kmeans.cluster_centers_))
dist_h[np.eye(dist_h.shape[0]) == 1] = np.nan

# Plot
f, ax = plt.subplots(1, 3, figsize=(15, 4))
sns.heatmap(dist_mni, ax=ax[0], square=True)
ax[0].set_title('Distance (MNI)')
sns.heatmap(dist_h, ax=ax[1], square=True)
ax[1].set_title('Distance (Hierarchy)')
my_regplot(dist_mni, dist_h, 'Distance (MNI)', 'Distance (Hierarchy)', ax[2])
f.savefig(os.path.join(environment.figdir, 'distance_vs_distance.png'), dpi=150, bbox_inches='tight', pad_inches=0.1)

# %% 2) Energy from group average adjacency matrix
# Parameters
control = 'minimum'
T = 1
B_ver = 'x0xfwb'

# Load energy (group A matrix; 6% sparsity)
file_label = 'disc_mean_A_s6_{0}_T-{1}_B-{2}-g{3}_E.npy'.format(control, T, B_ver, cg.kmeans.n_clusters)
print(file_label)

# Create DataMatrix with E_Am as data
E_Am = np.load(os.path.join(environment.pipelinedir, file_label))
E_Am = np.mean(E_Am, axis=2)
E_Am[np.eye(E_Am.shape[0]) == 1] = np.nan
E_Am = DataMatrix(data=E_Am)

# Plot
f, ax = plt.subplots(1, 1, figsize=(6, 5))
sns.heatmap(E_Am.data, ax=ax, square=True)
f.savefig(os.path.join(environment.figdir, 'meanadj_energy.png'), dpi=150, bbox_inches='tight', pad_inches=0.1)

# Plot
f, ax = plt.subplots(1, 4, figsize=(20, 4))
my_regplot(dist_mni, E_Am.data, 'Distance (MNI)', 'Energy', ax[0])
my_regplot(dist_h, E_Am.data, 'Distance (hierarchy)', 'Energy', ax[1])

E_Am.regress_nuisance(c=dist_h)
my_regplot(dist_mni, E_Am.data_resid, 'Distance (MNI)', 'Energy (resid hierarchy)', ax[2])
E_Am.regress_nuisance(c=dist_mni)
my_regplot(dist_h, E_Am.data_resid, 'Distance (hierarchy)', 'Energy (resid MNI)', ax[3])

f.subplots_adjust(wspace=0.5)
f.savefig(os.path.join(environment.figdir, 'meanadj_energy_vs_distance.png'), dpi=150, bbox_inches='tight',
          pad_inches=0.1)

# %% 3) Gradient traversal variance
filters = {'healthExcludev2': 0, 't1Exclude': 0,
           'b0ProtocolValidationStatus': 1, 'dti64ProtocolValidationStatus': 1, 'dti64Exclude': 0,
           'psychoactiveMedPsychv2': 0}
environment.load_metadata(filters)

# retain half as discovery set
environment.df['disc_repl'] = get_disc_repl(environment.df, frac=0.5)
environment.df = environment.df.loc[environment.df['disc_repl'] == 0, :]
print(environment.df.shape)

# Load sc data
loadsc = LoadSC(environment=environment, Subject=Subject)
loadsc.run()
A = loadsc.A
n_subs = loadsc.df.shape[0]

# Get streamline count and network density
A_c = np.zeros((n_subs,))
A_d = np.zeros((n_subs,))
for i in range(n_subs):
    A_c[i] = np.sum(np.triu(A[:, :, i]))
    A_d[i] = np.count_nonzero(np.triu(A[:, :, i])) / ((A[:, :, i].shape[0] ** 2 - A[:, :, i].shape[0]) / 2)

# Get group average adj. matrix
mean_spars = np.round(A_d.mean(), 2)
print(mean_spars)

A = np.mean(A, 2)
thresh = np.percentile(A, 100 - (mean_spars * 100))
A[A < thresh] = 0
print(np.count_nonzero(np.triu(A)) / ((A.shape[0] ** 2 - A.shape[0]) / 2))
A = DataMatrix(data=A)
A.get_gradient_variance(cg.gradients)

# Plot: gradient traversal vs. energy
x_to_plot = [A.hops, A.tm_var, A.smv_var, A.joint_var]
xlabels = ['Hops in sp', 'Transmodal variance', 'Unimodal variance', 'Joint variance (euclid)']
# E_Am.regress_nuisance(c=dist_mni); ylabel = 'Energy (resid MNI)'
E_Am.regress_nuisance(c=mean_over_clusters(A.hops, cg.kmeans.labels_))
ylabel = 'Energy (resid hops)'

f, ax = plt.subplots(1, len(x_to_plot), figsize=(len(x_to_plot) * 5, 4))
for i in range(len(x_to_plot)):
    my_regplot(mean_over_clusters(x_to_plot[i], cg.kmeans.labels_),
               E_Am.data_resid,
               xlabels[i], ylabel, ax[i])

f.subplots_adjust(wspace=0.5)
f.savefig(os.path.join(environment.figdir, 'meanadj_energy_vs_adjstats.png'), dpi=150, bbox_inches='tight',
          pad_inches=0.1)

# Plot: gradient traversal vs. space
x_to_plot = [mean_over_clusters(A.hops, cg.kmeans.labels_), dist_mni, dist_h]
xlabels = ['Hops in sp', 'Distance (MNI)', 'Distance (hierarchy)']
y_to_plot = [A.tm_var, A.smv_var, A.joint_var]
ylabels = ['Transmodal variance', 'Unimodal variance', 'Joint variance (euclid)']

f, ax = plt.subplots(len(x_to_plot), len(y_to_plot), figsize=(len(x_to_plot) * 5, len(y_to_plot) * 5))
for i in range(len(x_to_plot)):
    for j in range(len(y_to_plot)):
        my_regplot(x_to_plot[i],
                   mean_over_clusters(y_to_plot[j], cg.kmeans.labels_),
                   xlabels[i], ylabels[j], ax[i, j])

f.subplots_adjust(wspace=0.5)
f.savefig(os.path.join(environment.figdir, 'variance_vs_distance.png'), dpi=150, bbox_inches='tight', pad_inches=0.1)

# %% 4) Null network models
surr_type = 'spatial_wssp'  # 'standard' 'spatial_wwp' 'spatial_wsp' 'spatial_wssp'
# surr_type = surr_type+'_hybrid'
# surr_type = surr_type+'_grad_cmni'
surr_type = surr_type + '_mni_cgrad'

tm_var_surr = np.load(
    os.path.join(environment.pipelinedir, 'disc_mean_A_s6_{0}_grad{1}_tm_var_surr.npy'.format(surr_type, cg.n_clusters)))
smv_var_surr = np.load(
    os.path.join(environment.pipelinedir, 'disc_mean_A_s6_{0}_grad{1}_smv_var_surr.npy'.format(surr_type, cg.n_clusters)))
joint_var_surr = np.load(os.path.join(environment.pipelinedir,
                                      'disc_mean_A_s6_{0}_grad{1}_joint_var_surr.npy'.format(surr_type, cg.n_clusters)))

# E_Am.regress_nuisance(c=dist_mni); ylabel = 'Energy (resid MNI)'
E_Am.regress_nuisance(c=mean_over_clusters(A.hops, cg.kmeans.labels_))
ylabel = 'Energy (resid hops)'

f, ax = plt.subplots(1, 2, figsize=(10, 5))
my_nullplot(mean_over_clusters(A.tm_var, cg.kmeans.labels_), tm_var_surr, E_Am.data, 'null',
            ax=ax[0])
my_nullplot(mean_over_clusters(A.smv_var, cg.kmeans.labels_), smv_var_surr, E_Am.data, 'null',
            ax=ax[1])

f.subplots_adjust(wspace=0.5)
f.savefig(os.path.join(environment.figdir, 'meanadj_energy_vs_adjstats_null.png'), dpi=150, bbox_inches='tight',
          pad_inches=0.1)

# %% 5) bold rlfp
filters = {'healthExcludev2': 0, 't1Exclude': 0,
           'b0ProtocolValidationStatus': 1, 'dti64ProtocolValidationStatus': 1, 'dti64Exclude': 0,
           'psychoactiveMedPsychv2': 0, 'restProtocolValidationStatus': 1, 'restExclude': 0}
environment.load_metadata(filters)

# retain half as discovery set
environment.df['disc_repl'] = get_disc_repl(environment.df, frac=0.5)
environment.df = environment.df.loc[environment.df['disc_repl'] == 0, :]
print(environment.df.shape)

# Load rlfp data
loadrlfp = LoadRLFP(environment=environment, Subject=Subject)
loadrlfp.run()
rlfp = loadrlfp.rlfp
n_subs = loadrlfp.df.shape[0]

# mean over subjects
rlfp_mean = DataVector(data=np.mean(rlfp, axis=0))
rlfp_mean.regress_nuisance(c=A.hops.mean(axis=0))

# Plot: gradient traversal vs. rlfp
x_to_plot = [A.hops.mean(axis=0), A.tm_var.mean(axis=0), A.smv_var.mean(axis=0), A.joint_var.mean(axis=0)]
xlabels = ['Hops in sp', 'Transmodal variance', 'Unimodal variance', 'Joint variance (euclid)']

f, ax = plt.subplots(1, len(x_to_plot), figsize=(len(x_to_plot) * 5, 4))
for i in range(len(x_to_plot)):
    my_regplot(x_to_plot[i], rlfp_mean.data_resid, xlabels[i], 'RLFP (mean)', ax[i])

f.subplots_adjust(wspace=0.5)
f.savefig(os.path.join(environment.figdir, 'rlfp_vs_adjstats.png'), dpi=150, bbox_inches='tight',
          pad_inches=0.1)

# %% 6) gene co-expression
allen_expression_lh = np.load('/Volumes/work_ssd/research_data/allen/allen_expression_lh.npy')
allen_expression_rh = np.load('/Volumes/work_ssd/research_data/allen/allen_expression_rh.npy')
allen_expression = np.zeros(allen_expression_lh.shape)
allen_expression[:200, :] = allen_expression_lh[:200, :]
allen_expression[200:, :] = allen_expression_rh[200:, :]

allen_coexpression = DataMatrix(data=np.corrcoef(allen_expression, rowvar=True))
allen_coexpression.regress_nuisance(c=A.hops)

indices = np.where(~np.eye(n_parcels, dtype=bool) * ~np.isnan(allen_coexpression.data))
print(sp.stats.pearsonr(A.tm_var[indices], allen_coexpression.data[indices]))
print(sp.stats.pearsonr(A.tm_var[indices], allen_coexpression.data_resid[indices]))

# Plot
f, ax = plt.subplots(1, 2, figsize=(10, 5))
allen_coexpression.mean_over_clusters(cluster_labels=cg.kmeans.labels_, use_resid_matrix=True)
my_regplot(mean_over_clusters(A.tm_var, cg.kmeans.labels_),
           allen_coexpression.data_clusters,
           'Transmodal variance', 'Gene coexpression', ax[0])
my_regplot(mean_over_clusters(A.smv_var, cg.kmeans.labels_),
           allen_coexpression.data_clusters,
           'Unimodal variance', 'Gene coexpression', ax[1])

f.subplots_adjust(wspace=0.5)
f.savefig(os.path.join(environment.figdir, 'gene_vs_adjstats.png'), dpi=150, bbox_inches='tight',
          pad_inches=0.1)

# Plot nulls
f, ax = plt.subplots(1, 2, figsize=(10, 5))
my_nullplot(mean_over_clusters(A.tm_var, cg.kmeans.labels_), tm_var_surr,
            allen_coexpression.data_clusters, 'null', ax=ax[0])
my_nullplot(mean_over_clusters(A.smv_var, cg.kmeans.labels_), smv_var_surr,
            allen_coexpression.data_clusters, 'null', ax=ax[1])

f.subplots_adjust(wspace=0.5)
f.savefig(os.path.join(environment.figdir, 'gene_vs_adjstats_null.png'), dpi=150, bbox_inches='tight',
          pad_inches=0.1)
