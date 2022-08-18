import sys, os, platform
from src.pnc import Environment
from src.imaging_derivs import DataVector
from src.plotting import roi_to_vtx
from src.utils import get_states_from_brain_map
import numpy as np

# %% plotting
import seaborn as sns
import matplotlib.pyplot as plt
from nilearn import plotting
from src.plotting import set_plotting_params
set_plotting_params(format='svg')
figsize = 1.5

# %% Setup project environment
computer = 'macbook'
parc = 'schaefer'
n_parcels = 200
sc_edge_weight = 'streamlineCount'
environment = Environment(computer=computer, parc=parc, n_parcels=n_parcels, sc_edge_weight=sc_edge_weight)
environment.make_output_dirs()
environment.load_parc_data()

# %% get states
bbw_dir = os.path.join(environment.research_data, 'BigBrainWarp', 'spaces', 'fsaverage')
if parc == 'schaefer':
    state_brain_map = np.loadtxt(os.path.join(bbw_dir, 'Hist_G2_Schaefer2018_{0}Parcels_17Networks.txt' \
                                              .format(n_parcels)))
elif parc == 'glasser':
    state_brain_map = np.loadtxt(os.path.join(bbw_dir, 'Hist_G2_HCP-MMP1.txt'))
state_brain_map = state_brain_map * -1

bin_size = 10
n_states = int(n_parcels / bin_size)
states = get_states_from_brain_map(brain_map=state_brain_map, n_bins=n_states)

# %%
state_brain_map = DataVector(data=state_brain_map, name='state_brain_map')
state_brain_map.rankdata()
state_brain_map.rescale_unit_interval()

# %% plot
cmap = 'viridis'
figwidth = 1
figratio = 0.60
figheight = figwidth * figratio

for hemi in ['left', 'right']:
    if hemi == 'left':
        vtx_data, plot_min, plot_max = roi_to_vtx(state_brain_map.data + 1e-5, environment.parcel_names, environment.lh_annot_file)
        vtx_data = vtx_data.astype(float)
    elif hemi == 'right':
        vtx_data, plot_min, plot_max = roi_to_vtx(state_brain_map.data + 1e-5, environment.parcel_names, environment.rh_annot_file)
        vtx_data = vtx_data.astype(float)

    for view in ['lateral', 'medial']:
        f = plotting.plot_surf_roi(environment.fsaverage['infl_{0}'.format(hemi)], roi_map=vtx_data,
                                   hemi=hemi, view=view, vmin=0, vmax=1,
                                   bg_map=environment.fsaverage['sulc_{0}'.format(hemi)], bg_on_data=True,
                                   darkness=.5, cmap=cmap, colorbar=False)
        f.set_figwidth(figwidth)
        f.set_figheight(figheight)
        plt.subplots_adjust(0, 0, 1, 1, 0, 0)
        for ax in f.axes:
            ax.axis('off')
            ax.margins(x=0, y=0)
            ax.xaxis.set_major_locator(plt.NullLocator())
            ax.yaxis.set_major_locator(plt.NullLocator())
        f.savefig(os.path.join(environment.figdir, '{0}_{1}_{2}.png'.format(state_brain_map.name, hemi, view)),
                  dpi=1000, bbox_inches='tight',
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

# %% plot states separately
cmap = plt.get_cmap('Pastel1')
figwidth = .5
figratio = 0.60
figheight = figwidth * figratio

x = ([2, 1], [15, 2])
for i in np.arange(len(x)):
    states_ij = np.zeros(states.shape)
    states_ij[states == x[i][0]] = x[i][1]

    for hemi in ['left', 'right']:
        if hemi == 'left':
            vtx_data, plot_min, plot_max = roi_to_vtx(states_ij, environment.parcel_names, environment.lh_annot_file)
            vtx_data = vtx_data.astype(float)
        elif hemi == 'right':
            vtx_data, plot_min, plot_max = roi_to_vtx(states_ij, environment.parcel_names, environment.rh_annot_file)
            vtx_data = vtx_data.astype(float)

        for view in ['lateral', 'medial']:
            f = plotting.plot_surf_roi(environment.fsaverage['infl_{0}'.format(hemi)], roi_map=vtx_data,
                                       hemi=hemi, view=view, vmin=1, vmax=9,
                                       bg_map=environment.fsaverage['sulc_{0}'.format(hemi)], bg_on_data=True,
                                       darkness=.5, cmap=cmap, colorbar=False)
            f.set_figwidth(figwidth)
            f.set_figheight(figheight)
            f.savefig(os.path.join(environment.figdir, '{0}_{1}_{2}_{3}.png'.format('state', i, hemi, view)), dpi=1000,
                      bbox_inches='tight',
                      pad_inches=0)
            plt.close()

# %% dummy energy matrix
n_states = 5
indices_lower = np.tril_indices(n_states, k=-1)
figsize = 1.25

f, ax = plt.subplots(1, 1, figsize=(figsize, figsize))
np.random.seed(0)
x = np.random.rand(n_states, n_states)
x[np.eye(n_states) == 1] = np.nan
h = sns.heatmap(x, vmin=0, vmax=1, cmap='Purples', square=True,
                cbar_kws={'shrink': 0.8, 'label': 'energy (a.u.)'})
ax.set_title('e')
ax.set_ylabel("Initial states")
ax.set_xlabel("Target states")
cbar = ax.collections[0].colorbar
cbar.set_ticks([])
ax.tick_params(pad=-2.5)
f.savefig(os.path.join(environment.figdir, 'schematic_energy'), dpi=300, bbox_inches='tight',
          pad_inches=0.01)
plt.close()

f, ax = plt.subplots(1, 1, figsize=(figsize, figsize))
x = x.transpose() - x
plot_mask = np.zeros((n_states, n_states))
plot_mask[indices_lower] = 1
plot_mask = plot_mask.astype(bool)

h = sns.heatmap(x, mask=plot_mask, vmin=-1, vmax=1, cmap='coolwarm', square=True,
                cbar_kws={'shrink': 0.8, 'label': 'energy asymmetry'})
ax.set_title('ed')
ax.set_ylabel("Initial states")
ax.set_xlabel("Target states")
cbar = ax.collections[0].colorbar
cbar.set_ticks([])
ax.tick_params(pad=-2.5)
f.savefig(os.path.join(environment.figdir, 'schematic_energy_asym'), dpi=300, bbox_inches='tight',
          pad_inches=0.01)
plt.close()
