# Import
import os
import numpy as np
from brainspace.gradient import GradientMaps
import nibabel as nib
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
import seaborn as sns
from utils.plotting import roi_to_vtx
from nilearn import datasets
from nilearn import plotting

sns.set(style='white', context='talk', font_scale=1)
import matplotlib.font_manager as font_manager
fontpath = '/Users/lindenmp/Library/Fonts/PublicSans-Thin.ttf'
prop = font_manager.FontProperties(fname=fontpath)
plt.rcParams['font.family'] = prop.get_name()
plt.rcParams['svg.fonttype'] = 'none'

# Setup project environment
from data_loader.pnc import Environment, Subject
parc='schaefer'
n_parcels=400
sc_edge_weight='streamlineCount'
environment = Environment(parc=parc, n_parcels=n_parcels, sc_edge_weight=sc_edge_weight)
environment.make_output_dirs()

filters = {'healthExcludev2': 0, 't1Exclude': 0,
           'b0ProtocolValidationStatus': 1, 'dti64ProtocolValidationStatus': 1, 'dti64Exclude': 0,
           'psychoactiveMedPsychv2': 0, 'restProtocolValidationStatus': 1, 'restExclude': 0}
df = environment.load_metadata(filters)
print(df.shape)

# Load FC data
print('Loading resting state FC data')
n_subs = df.shape[0]
fc = np.zeros((n_parcels, n_parcels, n_subs))

for i in np.arange(n_subs):
    subject = Subject(df.index[i][0], df.index[i][1])
    subject.load_rsfc()
    fc[:, :, i] = subject.rsfc.copy()

# Average over subjects
pnc_conn_mat = np.nanmean(fc, axis=2)
pnc_conn_mat[np.eye(n_parcels, dtype=bool)] = 0
# pnc_conn_mat = dominant_set(pnc_conn_mat, 0.10, as_sparse = False)

# Plot mean fc matrix
f, ax = plt.subplots(1, figsize=(5, 5))
sns.heatmap(pnc_conn_mat, cmap='coolwarm', center=0, square=True)
ax.tick_params(pad = -2.5)
f.savefig(os.path.join(environment.figdir, 'mean_fc.png'), dpi=300, bbox_inches='tight')

# Generate gradients
gm_template = GradientMaps(n_components=2, approach='dm', kernel=None, random_state=0)
gm_template.fit(pnc_conn_mat)

# Plot eigenvalues
f, ax = plt.subplots(1, figsize=(5, 4))
ax.scatter(range(gm_template.lambdas_.size), gm_template.lambdas_)
ax.set_xlabel('Component Nb')
ax.set_ylabel('Eigenvalue')
ax.tick_params(pad = -2.5)
f.savefig(os.path.join(environment.figdir, 'gradient_eigenvals.png'), dpi=300, bbox_inches='tight')

if n_parcels == 200:
    gm_template.gradients_ = gm_template.gradients_ * -1
    gradients = np.zeros(gm_template.gradients_.shape)
    gradients[:,0], gradients[:,1] = gm_template.gradients_[:,1], gm_template.gradients_[:,0]
elif n_parcels == 400:
    gradients = np.zeros(gm_template.gradients_.shape)
    gradients[:,0], gradients[:,1] = gm_template.gradients_[:,1] * -1, gm_template.gradients_[:,0]
else:
    gm_template.gradients_ = gm_template.gradients_ * -1
    gradients = np.zeros(gm_template.gradients_.shape)
    gradients[:,0], gradients[:,1] = gm_template.gradients_[:,1], gm_template.gradients_[:,0]

np.savetxt(os.path.join(environment.outputdir,'gradients.txt'), gradients)

# Plot first two gradients
if parc == 'schaefer':
    fsaverage = datasets.fetch_surf_fsaverage(mesh='fsaverage5')
    parcel_names = np.genfromtxt(os.path.join(environment.projdir,
                                              'figs_support', 'labels',
                                              'schaefer{0}NodeNames.txt'.format((n_parcels))), dtype='str')
elif parc == 'glasser':
    fsaverage = datasets.fetch_surf_fsaverage(mesh='fsaverage')

for g in np.arange(0, 2):
    f, ax = plt.subplots(1, 4, figsize=(20, 5), subplot_kw={'projection': '3d'})
    plt.subplots_adjust(wspace=0, hspace=0)

    if parc == 'schaefer':
        parc_file = os.path.join(environment.projdir, 'figs_support', 'Parcellations', 'FreeSurfer5.3',
                                 'fsaverage5', 'label',
                                 'lh.Schaefer2018_{0}Parcels_17Networks_order.annot'.format(n_parcels))
    elif parc == 'glasser':
        parc_file = os.path.join(environment.projdir, 'figs_support', 'Parcellations', 'FreeSurfer5.3',
                                 'fsaverage', 'label', 'lh.HCP-MMP1.annot')

    labels, ctab, surf_names = nib.freesurfer.read_annot(parc_file)
    vtx_data, plot_min, plot_max = roi_to_vtx(gradients[:, g], parcel_names, parc_file)
    vtx_data = vtx_data.astype(float)
    plotting.plot_surf_roi(fsaverage['infl_left'], roi_map=vtx_data,
                           hemi='left', view='lateral', vmin=plot_min, vmax=plot_max,
                           bg_map=fsaverage['sulc_left'], bg_on_data=False, axes=ax[0],
                           darkness=.5, cmap='viridis');

    plotting.plot_surf_roi(fsaverage['infl_left'], roi_map=vtx_data,
                           hemi='left', view='medial', vmin=plot_min, vmax=plot_max,
                           bg_map=fsaverage['sulc_left'], bg_on_data=False, axes=ax[1],
                           darkness=.5, cmap='viridis');

    if parc == 'schaefer':
        parc_file = os.path.join(environment.projdir, 'figs_support', 'Parcellations', 'FreeSurfer5.3',
                                 'fsaverage5', 'label',
                                 'rh.Schaefer2018_{0}Parcels_17Networks_order.annot'.format(n_parcels))
    elif parc == 'glasser':
        parc_file = os.path.join(environment.projdir, 'figs_support', 'Parcellations', 'FreeSurfer5.3',
                                 'fsaverage', 'label', 'rh.HCP-MMP1.annot')

    labels, ctab, surf_names = nib.freesurfer.read_annot(parc_file)
    vtx_data, plot_min, plot_max = roi_to_vtx(gradients[:, g], parcel_names, parc_file)
    vtx_data = vtx_data.astype(float)
    plotting.plot_surf_roi(fsaverage['infl_right'], roi_map=vtx_data,
                           hemi='right', view='lateral', vmin=plot_min, vmax=plot_max,
                           bg_map=fsaverage['sulc_right'], bg_on_data=False, axes=ax[2],
                           darkness=.5, cmap='viridis');

    plotting.plot_surf_roi(fsaverage['infl_right'], roi_map=vtx_data,
                           hemi='right', view='medial', vmin=plot_min, vmax=plot_max,
                           bg_map=fsaverage['sulc_right'], bg_on_data=False, axes=ax[3],
                           darkness=.5, cmap='viridis');

    f.suptitle('Gradient ' + str(g + 1))
    f.savefig(os.path.join(environment.figdir, 'gradient_{0}.png'.format(g)), dpi=150, bbox_inches='tight', pad_inches=0)

# Cluster gradient
n_clusters=int(n_parcels*.05)
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(gradients)
unique, counts = np.unique(kmeans.labels_, return_counts=True)

# Plot clustered gradient
f, ax = plt.subplots(figsize=(5, 5))
ax.scatter(gradients[:,1], gradients[:,0], c=kmeans.labels_, cmap='Set3')
for i, txt in enumerate(np.arange(n_clusters)):
    ax.annotate(txt, (kmeans.cluster_centers_[i,1], kmeans.cluster_centers_[i,0]), ha="center", va="center", size=15)
ax.set_xlabel('Gradient 2')
ax.set_ylabel('Gradient 1')
ax.tick_params(pad = -2.5)
f.savefig(os.path.join(environment.figdir, 'gradient_clusters.png'), dpi=150, bbox_inches='tight', pad_inches=0.1)
