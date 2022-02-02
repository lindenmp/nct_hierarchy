import os
from pfactor_gradients.pnc import Environment, Subject
from pfactor_gradients.routines import LoadSC, LoadCT, LoadCBF, LoadAverageBrainMaps, LoadSA
from pfactor_gradients.pipelines import ComputeGradients
from pfactor_gradients.hcp import BrainMapLoader
from pfactor_gradients.imaging_derivs import DataVector
from pfactor_gradients.utils import get_null_p, get_states_from_brain_map
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
parc = 'glasser'
n_parcels = 360
sc_edge_weight = 'streamlineCount'
environment = Environment(computer=computer, parc=parc, n_parcels=n_parcels, sc_edge_weight=sc_edge_weight)
environment.make_output_dirs()
environment.load_parc_data()

# filter subjects
filters = {'healthExcludev2': 0, 'psychoactiveMedPsychv2': 0,
           't1Exclude': 0, 'fsFinalExclude': 0,
           'b0ProtocolValidationStatus': 1, 'dti64ProtocolValidationStatus': 1, 'dti64Exclude': 0}
           # 'restProtocolValidationStatus': 1, 'restExclude': 0} # need to add these filters in if doing funcg1 below
environment.load_metadata(filters)

# %% Load sc data
# note, this performs more subject exclusion
load_sc = LoadSC(environment=environment, Subject=Subject)
load_sc.run()
# refilter environment due to LoadSC excluding on disconnected nodes
environment.df = load_sc.df.copy()
n_subs = environment.df.shape[0]

# %% load mean brain maps
loaders_dict = {
    'ct': LoadCT(environment=environment, Subject=Subject),
    'sa': LoadSA(environment=environment, Subject=Subject)
}

load_average_bms = LoadAverageBrainMaps(loaders_dict=loaders_dict)
load_average_bms.run(return_descending=False)

# %% compute functional gradient
compute_gradients = ComputeGradients(environment=environment, Subject=Subject)
compute_gradients.run()

# append fc gradient to brain maps
dv = DataVector(data=compute_gradients.gradients[:, 0], name='func-g1')
dv.rankdata()
dv.rescale_unit_interval()
load_average_bms.brain_maps[dv.name] = dv

# %% get states
which_brain_map = 'hist-g2'

if which_brain_map == 'hist-g2':
    if computer == 'macbook':
        bbw_dir = '/Volumes/T7/research_data/BigBrainWarp/spaces/fsaverage/'
    elif computer == 'cbica':
        bbw_dir = '/cbica/home/parkesl/research_data/BigBrainWarp/spaces/fsaverage/'

    if parc == 'schaefer':
        state_brain_map = np.loadtxt(os.path.join(bbw_dir, 'Hist_G2_Schaefer2018_{0}Parcels_17Networks.txt'.format(n_parcels)))
    elif parc == 'glasser':
        state_brain_map = np.loadtxt(os.path.join(bbw_dir, 'Hist_G2_HCP-MMP1.txt'))
    state_brain_map = state_brain_map * -1
elif which_brain_map == 'func-g1':
    state_brain_map = compute_gradients.gradients[:, 0].copy()
else:
    state_brain_map = load_average_bms.brain_maps[which_brain_map].data.copy()

n_bins = int(n_parcels/10)
states = get_states_from_brain_map(brain_map=state_brain_map, n_bins=n_bins)
n_states = len(np.unique(states))

mask = ~np.eye(n_states, dtype=bool)
indices = np.where(mask)
indices_upper = np.triu_indices(n_states, k=1)
indices_lower = np.tril_indices(n_states, k=-1)

# %% orthogonalize brain maps against state map
for key in load_average_bms.brain_maps:
    # load_average_bms.brain_maps[key].regress_nuisance(state_brain_map)
    # load_average_bms.brain_maps[key].data = load_average_bms.brain_maps[key].data_resid.copy()
    # load_average_bms.brain_maps[key].rankdata()
    # load_average_bms.brain_maps[key].rescale_unit_interval()
    print(key, sp.stats.pearsonr(state_brain_map, load_average_bms.brain_maps[key].data))

# # %% load mean brain maps from HCP
# if parc == 'schaefer':
#     order = 'lhrh'
# elif parc == 'glasser':
#     order = 'rhlh'
# hcp_brain_maps = BrainMapLoader(computer=computer)
# hcp_brain_maps.load_ct(lh_annot_file=environment.lh_annot_file, rh_annot_file=environment.rh_annot_file, order=order)
# hcp_brain_maps.load_myelin(lh_annot_file=environment.lh_annot_file, rh_annot_file=environment.rh_annot_file, order=order)
# hcp_brain_maps.load_ndi(lh_annot_file=environment.lh_annot_file, rh_annot_file=environment.rh_annot_file, order=order)
# hcp_brain_maps.load_odi(lh_annot_file=environment.lh_annot_file, rh_annot_file=environment.rh_annot_file, order=order)
#
# # correlations
# print('HCP, corr(ct,myelin)', sp.stats.spearmanr(hcp_brain_maps.ct, hcp_brain_maps.myelin))
# print('HCP, corr(myelin,ndi)', sp.stats.spearmanr(hcp_brain_maps.myelin, hcp_brain_maps.ndi))
# print('HCP, corr(ct,odi)', sp.stats.spearmanr(hcp_brain_maps.ct, hcp_brain_maps.odi))
# print('HCP, corr(odi,ndi)', sp.stats.spearmanr(hcp_brain_maps.odi, hcp_brain_maps.ndi))

# %% spatial correlation between brainmaps with spin test
state_brain_map = DataVector(data=state_brain_map, name=which_brain_map)
state_brain_map.rankdata()
state_brain_map.rescale_unit_interval()
state_brain_map.shuffle_data(shuffle_indices=environment.spun_indices)

for bm in loaders_dict:
    load_average_bms.brain_maps[bm].shuffle_data(shuffle_indices=environment.spun_indices)

def null_helper(map_x, map_y):
    observed = sp.stats.pearsonr(map_x.data, map_y.data)[0]
    null = np.zeros(environment.spun_indices.shape[1])

    for i in np.arange(environment.spun_indices.shape[1]):
        null[i] = sp.stats.spearmanr(map_x.data_shuf[:, i], map_y.data)[0]

    p_val = get_null_p(observed, null)

    return observed, null, p_val

# # ct vs sa
# observed, null, p_val = null_helper(load_average_bms.brain_maps['ct'], load_average_bms.brain_maps['sa'])
# f, ax = plt.subplots(1, 1, figsize=(figsize, figsize))
# my_null_plot(observed=observed, null=null, p_val=p_val, xlabel='corr(ct,sa)', ax=ax)
# f.savefig(os.path.join(environment.figdir, 'corr(pnc_ct,pnc_sa)'), dpi=300,
#           bbox_inches='tight', pad_inches=0.01)
# plt.close()

# state_brain_map vs metrics
metric = 'ct'
observed, null, p_val = null_helper(state_brain_map, load_average_bms.brain_maps[metric])
f, ax = plt.subplots(1, 1, figsize=(figsize, figsize))
my_null_plot(observed=observed, null=null, p_val=p_val, xlabel='corr(state_brain_map,{0})'.format(metric), ax=ax)
f.savefig(os.path.join(environment.figdir, 'corr(state_brain_map,pnc_{0})'.format(metric)), dpi=300,
          bbox_inches='tight', pad_inches=0.01)
plt.close()

metric = 'func-g1'
observed, null, p_val = null_helper(state_brain_map, load_average_bms.brain_maps[metric])
f, ax = plt.subplots(1, 1, figsize=(figsize, figsize))
my_null_plot(observed=observed, null=null, p_val=p_val, xlabel='corr(state_brain_map,{0})'.format(metric), ax=ax)
f.savefig(os.path.join(environment.figdir, 'corr(state_brain_map,pnc_{0})'.format(metric)), dpi=300,
          bbox_inches='tight', pad_inches=0.01)
plt.close()

# %% plot brain maps on surface
cmap = 'viridis'
figwidth = 1
figratio = 0.60
figheight = figwidth * figratio

metrics = ['state_brain_map',] + list(load_average_bms.brain_maps.keys())
for metric in metrics:
    if metric == 'state_brain_map':
        data = state_brain_map
    else:
        data = load_average_bms.brain_maps[metric]

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
