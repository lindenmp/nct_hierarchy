import os
import numpy as np
import scipy as sp
import pandas as pd
from pfactor_gradients.pnc import Environment, Subject
from pfactor_gradients.routines import LoadSC, LoadCT, LoadRLFP, LoadCBF, LoadREHO, LoadALFF,\
    LoadAverageBrainMaps, LoadAverageSC
from pfactor_gradients.imaging_derivs import DataMatrix, DataVector
from pfactor_gradients.pipelines import ComputeGradients
from pfactor_gradients.plotting import my_regplot
from pfactor_gradients.utils import get_pdist_clusters, get_fdr_p
from tqdm import tqdm
from scipy import stats

# %% Plotting
import matplotlib.pyplot as plt
import seaborn as sns
from pfactor_gradients.plotting import roi_to_vtx
import nibabel as nib
from nilearn import plotting
sns.set(style='white', context='paper', font_scale=1)
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

if parc == 'schaefer' and n_parcels == 400:
    spars_thresh = 0.06
elif parc == 'schaefer' and n_parcels == 200:
    spars_thresh = 0.12
load_average_sc = LoadAverageSC(load_sc=load_sc, spars_thresh=spars_thresh)
load_average_sc.run()
A = load_average_sc.A.copy()

# %% load mean brain maps
loaders_dict = {
    'ct': LoadCT(environment=environment, Subject=Subject),
    # 'rlfp': LoadRLFP(environment=environment, Subject=Subject),
    'cbf': LoadCBF(environment=environment, Subject=Subject),
    'reho': LoadREHO(environment=environment, Subject=Subject),
    'alff': LoadALFF(environment=environment, Subject=Subject)
}

load_average_bms = LoadAverageBrainMaps(loaders_dict=loaders_dict)
load_average_bms.run(return_descending=False)

n_states = len(np.unique(compute_gradients.grad_bins))
mask = ~np.eye(n_states, dtype=bool)
indices = np.where(mask)

# %% Fig 1a

cmap='viridis'
principal_gradient = DataVector(data=compute_gradients.gradients[:, 0])
principal_gradient.rescale_unit_interval()

vtx_data, plot_min, plot_max = roi_to_vtx(principal_gradient.data+1e-5, environment.parcel_names, environment.lh_annot_file)
vtx_data = vtx_data.astype(float)
f = plotting.plot_surf_roi(environment.fsaverage['infl_left'], roi_map=vtx_data,
                       hemi='left', view='lateral', vmin=0, vmax=1,
                       bg_map=environment.fsaverage['sulc_left'], bg_on_data=True,
                       darkness=.5, cmap=cmap, colorbar=False)
f.set_figwidth(2.5)
f.set_figheight(1.5)
f.savefig(os.path.join(environment.figdir, 'fig-1a_grad1_lateral.png'), dpi=300, bbox_inches='tight', pad_inches=0)
plt.close()

vtx_data, plot_min, plot_max = roi_to_vtx(principal_gradient.data+1e-5, environment.parcel_names, environment.lh_annot_file)
vtx_data = vtx_data.astype(float)
f = plotting.plot_surf_roi(environment.fsaverage['infl_left'], roi_map=vtx_data,
                       hemi='left', view='medial', vmin=0, vmax=1,
                       bg_map=environment.fsaverage['sulc_left'], bg_on_data=True,
                       darkness=.5, cmap=cmap, colorbar=True)
f.set_figwidth(2.5)
f.set_figheight(1.5)
f.savefig(os.path.join(environment.figdir, 'fig-1a_grad1_medial.png'), dpi=300, bbox_inches='tight', pad_inches=0)
plt.close()

# %% Fig 1b
# from scipy.linalg import svd
# u, s, vt = svd(A)
# c=1
# A_norm = A / (c + s[0]) - np.eye(n_parcels)
# A_norm[np.eye(n_parcels) == 1] = np.nan
A_norm = sp.stats.zscore(A)
# A_norm = (A_norm - np.min(A_norm)) / (np.max(A_norm) - np.min(A_norm))

idx = np.argsort(principal_gradient.data)
A_norm = A_norm[idx, :][:, idx]

f, ax = plt.subplots(1, 1, figsize=(2.5, 2.5))
plt.subplots_adjust(wspace=0, hspace=0)
sns.heatmap(A_norm, ax=ax, square=True)
ax.tick_params(pad=-2.5)
ax.set_xticks([])
ax.set_yticks([])
plt.show()
f.savefig(os.path.join(environment.figdir, 'fig-1b_pnc_meanadj.png'), dpi=300, bbox_inches='tight', pad_inches=0)
plt.close()
