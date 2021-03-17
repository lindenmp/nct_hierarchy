import os
import numpy as np
from sklearn.cluster import KMeans
import scipy as sp

import matplotlib.pyplot as plt
import seaborn as sns
from utils.imaging_derivs import DataMatrix
from utils.utils import get_pdist_clusters, get_disc_repl, mean_over_clusters
from utils.plotting import my_regplot, my_nullplot
from data_loader.routines import load_sc

#%% Set general plotting params
sns.set(style='white', context='talk', font_scale=1)
import matplotlib.font_manager as font_manager
fontpath = '/Users/lindenmp/Library/Fonts/PublicSans-Thin.ttf'
prop = font_manager.FontProperties(fname=fontpath)
plt.rcParams['font.family'] = prop.get_name()
plt.rcParams['svg.fonttype'] = 'none'

#%% Setup project environment
from data_loader.pnc import Environment, Subject
parc = 'schaefer'
n_parcels = 400
sc_edge_weight = 'streamlineCount'
environment = Environment(parc=parc, n_parcels=n_parcels, sc_edge_weight=sc_edge_weight)
environment.make_output_dirs()
environment.load_parc_data()

filters = {'healthExcludev2': 0, 't1Exclude': 0,
           'b0ProtocolValidationStatus': 1, 'dti64ProtocolValidationStatus': 1, 'dti64Exclude': 0,
           'psychoactiveMedPsychv2': 0}
df = environment.load_metadata(filters)
df['disc_repl'] = get_disc_repl(df, frac=0.5)

df = df.loc[df['disc_repl'] == 0, :]
print(df.shape)

environment.df = df
environment.Subject = Subject

# Load gradients
n_clusters=int(n_parcels*.05)
print(n_clusters)
gradients = np.loadtxt(os.path.join(environment.outputdir, 'gradients.txt'))
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(gradients)

#%%  1) mni/gradient distance between hierarchy clusters
dist_mni = get_pdist_clusters(environment.centroids.values, kmeans.labels_, method='median')
dist_mni[np.eye(dist_mni.shape[0]) == 1] = np.nan

dist_h = sp.spatial.distance.squareform(sp.spatial.distance.pdist(kmeans.cluster_centers_))
dist_h[np.eye(dist_h.shape[0]) == 1] = np.nan

# Plot
indices = np.where(~np.eye(n_clusters,dtype=bool))
f, ax = plt.subplots(1, 3, figsize=(15, 4))
sns.heatmap(dist_mni, ax=ax[0], square=True)
ax[0].set_title('Distance (MNI)')
sns.heatmap(dist_h, ax=ax[1], square=True)
ax[1].set_title('Distance (Hierarchy)')
my_regplot(dist_mni[indices], dist_h[indices], 'Distance (MNI)', 'Distance (Hierarchy)', ax[2])
f.savefig(os.path.join(environment.figdir, 'distance_vs_distance.png'), dpi=150, bbox_inches='tight', pad_inches=0.1)

#%% 2) Energy from group average adjacency matrix
# Parameters
control = 'minimum'
T = 1
B_ver = 'x0xfwb'
indices = np.where(~np.eye(n_clusters, dtype=bool))
print(len(indices[0]))

# Load energy (group A matrix; 6% sparsity)
file_label = 'disc_mean_A_s6_'+control+'_T-'+str(T)+'_B-'+B_ver+'-g'+str(n_clusters)+'_E.npy'
print(file_label)

# Create DataMatrix with E_Am as data
E_Am = np.load(os.path.join(environment.pipelinedir, file_label))
E_Am = np.mean(E_Am, axis = 2)
E_Am[np.eye(E_Am.shape[0]) == 1] = np.nan
E_Am = DataMatrix(data=E_Am)

# Plot
f, ax = plt.subplots(1, 1, figsize=(6, 5))
sns.heatmap(E_Am.data, ax=ax, square = True)
f.savefig(os.path.join(environment.figdir, 'meanadj_energy.png'), dpi=150, bbox_inches='tight', pad_inches=0.1)

# Plot
f, ax = plt.subplots(1, 4, figsize=(20, 4))
my_regplot(dist_mni[indices], E_Am.data[indices], 'Distance (MNI)', 'Energy', ax[0])
my_regplot(dist_h[indices], E_Am.data[indices], 'Distance (hierarchy)', 'Energy', ax[1])

E_Am.regress_nuisance(c=dist_h, indices=indices)
my_regplot(dist_mni[indices], E_Am.data_resid[indices], 'Distance (MNI)', 'Energy (resid hierarchy)', ax[2])
E_Am.regress_nuisance(c=dist_mni, indices=indices)
my_regplot(dist_h[indices], E_Am.data_resid[indices], 'Distance (hierarchy)', 'Energy (resid MNI)', ax[3])

f.subplots_adjust(wspace=0.5)
f.savefig(os.path.join(environment.figdir, 'meanadj_energy_vs_distance.png'), dpi=150, bbox_inches='tight', pad_inches=0.1)

#%% 3) Gradient traversal variance
# Load sc data
environment = load_sc(environment)
df, A = environment.df, environment.A
n_subs = df.shape[0]
del(environment.df, environment.A)

# Get streamline count and network density
A_c = np.zeros((n_subs,))
A_d = np.zeros((n_subs,))
for i in range(n_subs):
    A_c[i] = np.sum(np.triu(A[:, :, i]))
    A_d[i] = np.count_nonzero(np.triu(A[:, :, i]))/((A[:, :, i].shape[0]**2-A[:, :, i].shape[0])/2)

df['streamline_count'] = A_c
df['network_density'] = A_d

# Get group average adj. matrix
mean_spars = np.round(df['network_density'].mean(), 2)
print(mean_spars)

A = np.mean(A, 2)
thresh = np.percentile(A, 100 - (mean_spars * 100))
A[A < thresh] = 0
print(np.count_nonzero(np.triu(A)) / ((A.shape[0] ** 2 - A.shape[0]) / 2))
A = DataMatrix(data=A)
A.get_gradient_variance(gradients)

# Plot: gradient traversal vs. energy
x_to_plot = [A.hops, A.tm_var, A.smv_var, A.joint_var]
xlabels = ['Hops in sp', 'Transmodal variance', 'Unimodal variance', 'Joint variance (euclid)']
# E_Am.regress_nuisance(c=dist_mni, indices=indices); ylabel = 'Energy (resid MNI)'
E_Am.regress_nuisance(c=mean_over_clusters(A.hops, kmeans.labels_), indices=indices); ylabel = 'Energy (resid hops)'

f, ax = plt.subplots(1, len(x_to_plot), figsize=(len(x_to_plot)*5, 4))
for i in range(len(x_to_plot)):
    my_regplot(mean_over_clusters(x_to_plot[i], kmeans.labels_)[indices],
               E_Am.data_resid[indices],
               xlabels[i], ylabel, ax[i])

f.subplots_adjust(wspace=0.5)
f.savefig(os.path.join(environment.figdir, 'meanadj_energy_vs_adjstats.png'), dpi=150, bbox_inches='tight', pad_inches=0.1)

# Plot: gradient traversal vs. space
x_to_plot = [mean_over_clusters(A.hops, kmeans.labels_), dist_mni, dist_h]
xlabels = ['Hops in sp', 'Distance (MNI)', 'Distance (hierarchy)']
y_to_plot = [A.tm_var, A.smv_var, A.joint_var]
ylabels = ['Transmodal variance', 'Unimodal variance', 'Joint variance (euclid)']

f, ax = plt.subplots(len(x_to_plot), len(y_to_plot), figsize=(len(x_to_plot)*5, len(y_to_plot)*5))
for i in range(len(x_to_plot)):
    for j in range(len(y_to_plot)):
        my_regplot(x_to_plot[i][indices],
                   mean_over_clusters(y_to_plot[j], kmeans.labels_)[indices],
                   xlabels[i], ylabels[j], ax[i,j])

f.subplots_adjust(wspace=0.5)
f.savefig(os.path.join(environment.figdir, 'variance_vs_distance.png'), dpi=150, bbox_inches='tight', pad_inches=0.1)

#%% 4) Null network models
surr_type = 'spatial_wssp' # 'standard' 'spatial_wwp' 'spatial_wsp' 'spatial_wssp'
# surr_type = surr_type+'_hybrid'
# surr_type = surr_type+'_grad_cmni'
surr_type = surr_type+'_mni_cgrad'

tm_var_surr = np.load(os.path.join(environment.pipelinedir, 'disc_mean_A_s6_{0}_grad{1}_tm_var_surr.npy'.format(surr_type, n_clusters)))
smv_var_surr = np.load(os.path.join(environment.pipelinedir, 'disc_mean_A_s6_{0}_grad{1}_smv_var_surr.npy'.format(surr_type, n_clusters)))
joint_var_surr = np.load(os.path.join(environment.pipelinedir, 'disc_mean_A_s6_{0}_grad{1}_joint_var_surr.npy'.format(surr_type, n_clusters)))

# E_Am.regress_nuisance(c=dist_mni, indices=indices); ylabel = 'Energy (resid MNI)'
E_Am.regress_nuisance(c=mean_over_clusters(A.hops, kmeans.labels_), indices=indices); ylabel = 'Energy (resid hops)'

f, ax = plt.subplots(1, 2, figsize=(10, 5))
my_nullplot(mean_over_clusters(A.tm_var, kmeans.labels_)[indices], tm_var_surr[indices], E_Am.data[indices], 'null', ax=ax[0])
my_nullplot(mean_over_clusters(A.smv_var, kmeans.labels_)[indices], smv_var_surr[indices], E_Am.data[indices], 'null', ax=ax[1])

f.subplots_adjust(wspace=0.5)
f.savefig(os.path.join(environment.figdir, 'meanadj_energy_vs_adjstats_null.png'), dpi=150, bbox_inches='tight', pad_inches=0.1)
