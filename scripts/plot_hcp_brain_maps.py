import os
from pfactor_gradients.pnc import Environment
from pfactor_gradients.hcp import BrainMapLoader
from pfactor_gradients.imaging_derivs import DataVector
from pfactor_gradients.plotting import roi_to_vtx
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

# %% get a brain map
hcp_brain_maps = BrainMapLoader()
hcp_brain_maps.load_ct(lh_annot_file=environment.lh_annot_file, rh_annot_file=environment.rh_annot_file)
hcp_brain_maps.load_myelin(lh_annot_file=environment.lh_annot_file, rh_annot_file=environment.rh_annot_file)
hcp_brain_maps.load_ndi(lh_annot_file=environment.lh_annot_file, rh_annot_file=environment.rh_annot_file)
hcp_brain_maps.load_odi(lh_annot_file=environment.lh_annot_file, rh_annot_file=environment.rh_annot_file)

# %% correlations
print('corr(ct,myelin)', sp.stats.spearmanr(hcp_brain_maps.ct, hcp_brain_maps.myelin))
print('corr(myelin,ndi)', sp.stats.spearmanr(hcp_brain_maps.myelin, hcp_brain_maps.ndi))
print('corr(ct,odi)', sp.stats.spearmanr(hcp_brain_maps.ct, hcp_brain_maps.odi))
print('corr(odi,ndi)', sp.stats.spearmanr(hcp_brain_maps.odi, hcp_brain_maps.ndi))

# %% plot
metric = 'myelin'
if metric == 'ct':
    data = DataVector(data=hcp_brain_maps.ct, name='hcp_ct')
    data.rankdata()
    data.rescale_unit_interval()
elif metric == 'myelin':
    data = DataVector(data=hcp_brain_maps.myelin, name='hcp_myelin')
    data.rankdata()
    data.rescale_unit_interval()
elif metric == 'ndi':
    data = DataVector(data=hcp_brain_maps.ndi, name='hcp_ndi')
    data.rankdata()
    data.rescale_unit_interval()
elif metric == 'odi':
    data = DataVector(data=hcp_brain_maps.odi, name='hcp_odi')
    data.rankdata()
    data.rescale_unit_interval()

vtx_data, plot_min, plot_max = roi_to_vtx(data.data + 1e-5,
                                          environment.parcel_names, environment.lh_annot_file)
vtx_data = vtx_data.astype(float)

cmap = 'viridis'
figwidth = 1
figratio = 0.60
figheight = figwidth * figratio

f = plotting.plot_surf_roi(environment.fsaverage['infl_left'], roi_map=vtx_data,
                           hemi='left', view='lateral', vmin=0, vmax=1,
                           bg_map=environment.fsaverage['sulc_left'], bg_on_data=True,
                           darkness=.5, cmap=cmap, colorbar=False)
f.set_figwidth(figwidth)
f.set_figheight(figheight)
f.savefig(os.path.join(environment.figdir, '{0}_lat.png'.format(data.name)), dpi=300, bbox_inches='tight',
          pad_inches=0)
plt.close()

f = plotting.plot_surf_roi(environment.fsaverage['infl_left'], roi_map=vtx_data,
                           hemi='left', view='medial', vmin=0, vmax=1,
                           bg_map=environment.fsaverage['sulc_left'], bg_on_data=True,
                           darkness=.5, cmap=cmap, colorbar=False)
f.set_figwidth(figwidth)
f.set_figheight(figheight)
f.savefig(os.path.join(environment.figdir, '{0}_med.png'.format(data.name)), dpi=300, bbox_inches='tight',
          pad_inches=0)
plt.close()