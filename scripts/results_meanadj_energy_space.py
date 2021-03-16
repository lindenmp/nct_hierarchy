import os
import numpy as np
from sklearn.cluster import KMeans
import scipy as sp

import matplotlib.pyplot as plt
import seaborn as sns
from utils.imaging_derivs import DataMatrix
from utils.utils import get_pdist_clusters
from utils.plotting import my_regplot

#%% Set general plotting params
sns.set(style='white', context='talk', font_scale=1)
import matplotlib.font_manager as font_manager
fontpath = '/Users/lindenmp/Library/Fonts/PublicSans-Thin.ttf'
prop = font_manager.FontProperties(fname=fontpath)
plt.rcParams['font.family'] = prop.get_name()
plt.rcParams['svg.fonttype'] = 'none'

#%% Setup project environment
from data_loader.pnc import Environment
parc='schaefer'
n_parcels=400
sc_edge_weight='streamlineCount'
environment = Environment(parc=parc, n_parcels=n_parcels, sc_edge_weight=sc_edge_weight)
environment.make_output_dirs()
environment.load_parc_data()

#%%  1) mni/gradient distance between hierarchy clusters
n_clusters=int(n_parcels*.05)
print(n_clusters)
gradients = np.loadtxt(os.path.join(environment.outputdir, 'gradients.txt'))
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(gradients)

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
