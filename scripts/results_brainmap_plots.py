import os
from pfactor_gradients.pnc import Environment, Subject
from pfactor_gradients.routines import LoadSC, LoadCT, LoadCBF, LoadAverageBrainMaps
from pfactor_gradients.hcp import BrainMapLoader
from pfactor_gradients.imaging_derivs import DataVector
from pfactor_gradients.utils import get_null_p
from pfactor_gradients.plotting import my_null_plot, roi_to_vtx
import numpy as np
import pandas as pd
import scipy as sp

# %% Plotting
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
from nilearn import plotting
from pfactor_gradients.plotting import set_plotting_params
set_plotting_params(format='png')
figsize = 1.5

# %% Setup project environment
computer = 'macbook'
parc = 'schaefer'
n_parcels = 400
sc_edge_weight = 'streamlineCount'
environment = Environment(computer=computer, parc=parc, n_parcels=n_parcels, sc_edge_weight=sc_edge_weight)
environment.make_output_dirs()
environment.load_parc_data()

filters = {'healthExcludev2': 0, 't1Exclude': 0,
           'b0ProtocolValidationStatus': 1, 'dti64ProtocolValidationStatus': 1, 'dti64Exclude': 0,
           'psychoactiveMedPsychv2': 0, 'restProtocolValidationStatus': 1, 'restExclude': 0}
environment.load_metadata(filters)

# %% Load sc data
load_sc = LoadSC(environment=environment, Subject=Subject)
load_sc.run()
# refilter environment due to LoadSC excluding on disconnected nodes
environment.df = load_sc.df.copy()
n_subs = environment.df.shape[0]

# %% load mean brain maps from PNC
loaders_dict = {
    'ct': LoadCT(environment=environment, Subject=Subject),
    'cbf': LoadCBF(environment=environment, Subject=Subject)
}

load_average_bms = LoadAverageBrainMaps(loaders_dict=loaders_dict)
load_average_bms.run(return_descending=False)

# correlations
print('PNC, corr(ct,cbf)', sp.stats.spearmanr(load_average_bms.brain_maps['ct'].data,
                                         load_average_bms.brain_maps['cbf'].data))

# %% load mean brain maps from HCP
if parc == 'schaefer':
    order = 'lhrh'
elif parc == 'glasser':
    order = 'rhlh'
hcp_brain_maps = BrainMapLoader()
hcp_brain_maps.load_ct(lh_annot_file=environment.lh_annot_file, rh_annot_file=environment.rh_annot_file, order=order)
hcp_brain_maps.load_myelin(lh_annot_file=environment.lh_annot_file, rh_annot_file=environment.rh_annot_file, order=order)
hcp_brain_maps.load_ndi(lh_annot_file=environment.lh_annot_file, rh_annot_file=environment.rh_annot_file, order=order)
hcp_brain_maps.load_odi(lh_annot_file=environment.lh_annot_file, rh_annot_file=environment.rh_annot_file, order=order)

# correlations
print('HCP, corr(ct,myelin)', sp.stats.spearmanr(hcp_brain_maps.ct, hcp_brain_maps.myelin))
print('HCP, corr(myelin,ndi)', sp.stats.spearmanr(hcp_brain_maps.myelin, hcp_brain_maps.ndi))
print('HCP, corr(ct,odi)', sp.stats.spearmanr(hcp_brain_maps.ct, hcp_brain_maps.odi))
print('HCP, corr(odi,ndi)', sp.stats.spearmanr(hcp_brain_maps.odi, hcp_brain_maps.ndi))

# %% spatial correlation between brainmaps with spin test
load_average_bms.brain_maps['ct'].shuffle_data(shuffle_indices=environment.spun_indices)

# ct vs cbf
r = sp.stats.pearsonr(load_average_bms.brain_maps['ct'].data, load_average_bms.brain_maps['cbf'].data)[0]
null = np.zeros(environment.spun_indices.shape[1])
for i in np.arange(environment.spun_indices.shape[1]):
    null[i] = sp.stats.pearsonr(load_average_bms.brain_maps['ct'].data_shuf[:, i],
                                load_average_bms.brain_maps['cbf'].data)[0]

p_val = get_null_p(r, null)

f, ax = plt.subplots(1, 1, figsize=(figsize, figsize))
my_null_plot(observed=r, null=null, p_val=p_val, xlabel='corr(ct,cbf)', ax=ax)
f.savefig(os.path.join(environment.figdir, 'corr(pnc_ct,pnc_cbf)'), dpi=300,
          bbox_inches='tight', pad_inches=0.01)
plt.close()

# ct vs myelin
myelin = DataVector(data=hcp_brain_maps.myelin, name='hcp_myelin')
myelin.rankdata()
myelin.rescale_unit_interval()
r = sp.stats.pearsonr(load_average_bms.brain_maps['ct'].data, myelin.data)[0]
null = np.zeros(environment.spun_indices.shape[1])
for i in np.arange(environment.spun_indices.shape[1]):
    null[i] = sp.stats.pearsonr(load_average_bms.brain_maps['ct'].data_shuf[:, i],
                                myelin.data)[0]

p_val = get_null_p(r, null, version='reverse')

f, ax = plt.subplots(1, 1, figsize=(figsize, figsize))
my_null_plot(observed=r, null=null, p_val=p_val, xlabel='corr(ct,myelin)', ax=ax)
f.savefig(os.path.join(environment.figdir, 'corr(pnc_ct,hcp_myelin)'), dpi=300,
          bbox_inches='tight', pad_inches=0.01)
plt.close()

# %% plot brain maps on surface
cmap = 'viridis'
figwidth = 1
figratio = 0.60
figheight = figwidth * figratio

metrics = ['pnc_ct', 'pnc_cbf', 'hcp_ct', 'hcp_myelin', 'hcp_ndi', 'hcp_odi']
for metric in metrics:
    if metric == 'pnc_ct':
        data = DataVector(data=load_average_bms.brain_maps['ct'].data, name=metric)
    elif metric == 'pnc_cbf':
        data = DataVector(data=load_average_bms.brain_maps['cbf'].data, name=metric)
    elif metric == 'hcp_ct':
        data = DataVector(data=hcp_brain_maps.ct, name=metric)
        data.rankdata()
        data.rescale_unit_interval()
    elif metric == 'hcp_myelin':
        data = DataVector(data=hcp_brain_maps.myelin, name=metric)
        data.rankdata()
        data.rescale_unit_interval()
    elif metric == 'hcp_ndi':
        data = DataVector(data=hcp_brain_maps.ndi, name=metric)
        data.rankdata()
        data.rescale_unit_interval()
    elif metric == 'hcp_odi':
        data = DataVector(data=hcp_brain_maps.odi, name=metric)
        data.rankdata()
        data.rescale_unit_interval()

    # left hemisphere
    for hemi in ['left', 'right']:
        if hemi == 'left':
            vtx_data, plot_min, plot_max = roi_to_vtx(data.data + 1e-5, environment.parcel_names, environment.lh_annot_file)
        elif hemi == 'right':
            vtx_data, plot_min, plot_max = roi_to_vtx(data.data + 1e-5, environment.parcel_names, environment.rh_annot_file)
        vtx_data = vtx_data.astype(float)

        for view in ['lateral', 'medial']:
            f = plotting.plot_surf_roi(environment.fsaverage['infl_{0}'.format(hemi)], roi_map=vtx_data,
                                       hemi='{0}'.format(hemi), view='{0}'.format(view), vmin=0, vmax=1,
                                       bg_map=environment.fsaverage['sulc_{0}'.format(hemi)], bg_on_data=True,
                                       darkness=.5, cmap=cmap, colorbar=False)
            f.set_figwidth(figwidth)
            f.set_figheight(figheight)
            f.savefig(os.path.join(environment.figdir, '{0}_{1}_{2}'.format(data.name, hemi, view)),
                      dpi=300, bbox_inches='tight', pad_inches=0)
            plt.close()
