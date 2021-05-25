import sys, os, platform
from pfactor_gradients.pnc import Environment, Subject
from pfactor_gradients.pipelines import ComputeGradients
from pfactor_gradients.imaging_derivs import DataVector
from pfactor_gradients.plotting import my_regplot, my_nullplot, roi_to_vtx
import numpy as np
import pandas as pd
import scipy as sp

# %% Plotting
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
from nilearn import plotting

sns.set(style='white', context='paper', font_scale=1)
import matplotlib.font_manager as font_manager
fontpath = '/System/Library/Fonts/HelveticaNeue.ttc'
prop = font_manager.FontProperties(fname=fontpath)
prop.set_weight = 'thin'
plt.rcParams['font.family'] = prop.get_family()
plt.rcParams['font.sans-serif'] = prop.get_name()
# plt.rcParams['font.weight'] = 'thin'
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
n_bins = int(n_parcels/10)
compute_gradients = ComputeGradients(environment=environment, Subject=Subject, n_bins=n_bins)
compute_gradients.run()

n_states = len(np.unique(compute_gradients.grad_bins))
mask = ~np.eye(n_states, dtype=bool)
indices = np.where(mask)
indices_upper = np.triu_indices(n_states, k=1)
indices_lower = np.tril_indices(n_states, k=-1)

grad = DataVector(data=compute_gradients.gradients[:, 0], name='principal_gradient')
grad.rankdata()
grad.rescale_unit_interval()

states = compute_gradients.grad_bins

# %% plot
cmap = 'viridis'
figwidth = 1
figratio = 0.60
figheight = figwidth * figratio

vtx_data, plot_min, plot_max = roi_to_vtx(grad.data + 1e-5,
                                          environment.parcel_names, environment.lh_annot_file)
vtx_data = vtx_data.astype(float)

f = plotting.plot_surf_roi(environment.fsaverage['infl_left'], roi_map=vtx_data,
                           hemi='left', view='lateral', vmin=0, vmax=1,
                           bg_map=environment.fsaverage['sulc_left'], bg_on_data=True,
                           darkness=.5, cmap=cmap, colorbar=False)
f.set_figwidth(figwidth)
f.set_figheight(figheight)
plt.subplots_adjust(0,0,1,1,0,0)
for ax in f.axes:
    ax.axis('off')
    ax.margins(x=0, y=0)
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())
f.savefig(os.path.join(environment.figdir, '{0}_lat.png'.format(grad.name)), dpi=300, bbox_inches='tight',
          pad_inches=0)
plt.close()

f = plotting.plot_surf_roi(environment.fsaverage['infl_left'], roi_map=vtx_data,
                           hemi='left', view='medial', vmin=0, vmax=1,
                           bg_map=environment.fsaverage['sulc_left'], bg_on_data=True,
                           darkness=.5, cmap=cmap, colorbar=False)
f.set_figwidth(figwidth)
f.set_figheight(figheight)
plt.subplots_adjust(0,0,1,1,0,0)
for ax in f.axes:
    ax.axis('off')
    ax.margins(x=0, y=0)
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())
f.savefig(os.path.join(environment.figdir, '{0}_med.png'.format(grad.name)), dpi=300, bbox_inches='tight',
          pad_inches=0)
plt.close()

# %% colorbar
f, ax = plt.subplots(1, 1, figsize=(.5, .5))
h = sns.heatmap(np.zeros((5, 5)), vmin=0, vmax=1, cmap=cmap, cbar_kws={"orientation": "vertical"})
ax.set_xticklabels('')
ax.set_yticklabels('')
ax.remove()
cbar = ax.collections[0].colorbar
cbar.set_ticks([])
f.savefig(os.path.join(environment.figdir, 'colorbar_viridis.png'), dpi=300, bbox_inches='tight',
          pad_inches=0)
plt.close()

# %% plot states
cmap = plt.get_cmap('Spectral', np.max(states)-np.min(states)+1)
figwidth = 2.5
figratio = 0.60
figheight = figwidth * figratio

vtx_data, plot_min, plot_max = roi_to_vtx(states + 1,
                                          environment.parcel_names, environment.lh_annot_file)
vtx_data = vtx_data.astype(float)

f = plotting.plot_surf_roi(environment.fsaverage['infl_left'], roi_map=vtx_data,
                           hemi='left', view='lateral', vmin=plot_min, vmax=plot_max,
                           bg_map=environment.fsaverage['sulc_left'], bg_on_data=True,
                           darkness=.5, cmap=cmap, colorbar=True)
f.set_figwidth(figwidth)
f.set_figheight(figheight)
f.savefig(os.path.join(environment.figdir, 'states_lat.png'), dpi=300, bbox_inches='tight',
          pad_inches=0)
plt.close()

f = plotting.plot_surf_roi(environment.fsaverage['infl_left'], roi_map=vtx_data,
                           hemi='left', view='medial', vmin=plot_min, vmax=plot_max,
                           bg_map=environment.fsaverage['sulc_left'], bg_on_data=True,
                           darkness=.5, cmap=cmap, colorbar=True)
f.set_figwidth(figwidth)
f.set_figheight(figheight)
f.savefig(os.path.join(environment.figdir, 'states_med.png'), dpi=300, bbox_inches='tight',
          pad_inches=0)
plt.close()

# %% plot state pair
cmap = plt.get_cmap('Pastel1')
figwidth = 1
figratio = 0.60
figheight = figwidth * figratio

states_ij = np.zeros(states.shape)
states_ij[states == 1] = 1
states_ij[states == 32] = 2

vtx_data, plot_min, plot_max = roi_to_vtx(states_ij,
                                          environment.parcel_names, environment.lh_annot_file)
vtx_data = vtx_data.astype(float)

f = plotting.plot_surf_roi(environment.fsaverage['infl_left'], roi_map=vtx_data,
                           hemi='left', view='lateral', vmin=1, vmax=9,
                           bg_map=environment.fsaverage['sulc_left'], bg_on_data=True,
                           darkness=.5, cmap=cmap, colorbar=False)
f.set_figwidth(figwidth)
f.set_figheight(figheight)
f.savefig(os.path.join(environment.figdir, 'states_ij_lat.png'), dpi=300, bbox_inches='tight',
          pad_inches=0)
plt.close()

f = plotting.plot_surf_roi(environment.fsaverage['infl_left'], roi_map=vtx_data,
                           hemi='left', view='medial', vmin=1, vmax=9,
                           bg_map=environment.fsaverage['sulc_left'], bg_on_data=True,
                           darkness=.5, cmap=cmap, colorbar=False)
f.set_figwidth(figwidth)
f.set_figheight(figheight)
f.savefig(os.path.join(environment.figdir, 'states_ij_med.png'), dpi=300, bbox_inches='tight',
          pad_inches=0)
plt.close()

# %% separately
figwidth = .5
figratio = 0.60
figheight = figwidth * figratio

x = ([1, 1], [32, 2])
for i in np.arange(len(x)):
    states_ij = np.zeros(states.shape)
    states_ij[states == x[i][0]] = x[i][1]

    vtx_data, plot_min, plot_max = roi_to_vtx(states_ij,
                                              environment.parcel_names, environment.lh_annot_file)
    vtx_data = vtx_data.astype(float)

    f = plotting.plot_surf_roi(environment.fsaverage['infl_left'], roi_map=vtx_data,
                               hemi='left', view='lateral', vmin=1, vmax=9,
                               bg_map=environment.fsaverage['sulc_left'], bg_on_data=True,
                               darkness=.5, cmap=cmap, colorbar=False)
    f.set_figwidth(figwidth)
    f.set_figheight(figheight)
    f.savefig(os.path.join(environment.figdir, 'state_{0}_lat.png'.format(i)), dpi=300, bbox_inches='tight',
              pad_inches=0)
    plt.close()

    f = plotting.plot_surf_roi(environment.fsaverage['infl_left'], roi_map=vtx_data,
                               hemi='left', view='medial', vmin=1, vmax=9,
                               bg_map=environment.fsaverage['sulc_left'], bg_on_data=True,
                               darkness=.5, cmap=cmap, colorbar=False)
    f.set_figwidth(figwidth)
    f.set_figheight(figheight)
    f.savefig(os.path.join(environment.figdir, 'state_{0}_med.png'.format(i)), dpi=300, bbox_inches='tight',
              pad_inches=0)
    plt.close()

# %% dummy energy matrix
n_states = 5
indices_upper = np.triu_indices(n_states, k=1)
figsize = 1.25

f, ax = plt.subplots(1, 1, figsize=(figsize, figsize))
np.random.seed(0)
x = np.random.rand(n_states, n_states)
x[np.eye(n_states) == 1] = np.nan
# x = x + x.transpose()
h = sns.heatmap(x, vmin=0, vmax=1, cmap='Purples', square=True, cbar_kws={'shrink': 0.8, 'label': 'energy (a.u.)'})
ax.set_title('e')
ax.set_ylabel("Initial states")
ax.set_xlabel("Target states")
ax.set_yticklabels(['', 'i', '', 'j'])
ax.set_xticklabels(['', 'i', '', 'j'])
cbar = ax.collections[0].colorbar
cbar.set_ticks([])
ax.tick_params(pad=-2.5)
f.savefig(os.path.join(environment.figdir, 'schematic_energy.svg'), dpi=300, bbox_inches='tight',
          pad_inches=0.01)
plt.close()

f, ax = plt.subplots(1, 1, figsize=(figsize, figsize))
x = x.transpose() - x
plot_mask = np.zeros((n_states, n_states))
plot_mask[indices_upper] = 1
plot_mask = plot_mask.astype(bool)

h = sns.heatmap(x, mask=plot_mask, vmin=-1, vmax=1, cmap='coolwarm', square=True, cbar_kws={'shrink': 0.8, 'label': 'energy asymmetry'})
ax.set_title('ed')
ax.set_ylabel("Initial states")
ax.set_xlabel("Target states")
ax.set_yticklabels(['', '', '', 'i'])
ax.set_xticklabels(['', 'j', '', ''])
cbar = ax.collections[0].colorbar
cbar.set_ticks([])
ax.tick_params(pad=-2.5)
f.savefig(os.path.join(environment.figdir, 'schematic_energy_asym.svg'), dpi=300, bbox_inches='tight',
          pad_inches=0.01)
plt.close()