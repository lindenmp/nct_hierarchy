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

# %% get minimum control energy
n_subs = environment.df.shape[0]
n_states = len(np.unique(compute_gradients.grad_bins))
mask = ~np.eye(n_states, dtype=bool)
indices = np.where(mask)

df_energy = pd.DataFrame()

if parc == 'schaefer' and n_parcels == 400:
    sparse_thresh = 0.06
elif parc == 'schaefer' and n_parcels == 200:
    sparse_thresh = 0.12

my_list = list(load_average_bms.brain_maps.keys()) + ['wb']
for i in my_list:
    file = 'average_adj_n-{0}_s-{1}_ns-{2}-0_c-minimum_fast_T-1_B-{3}_E.npy'.format(n_subs, sparse_thresh, n_states, i)
    E = np.load(os.path.join(environment.pipelinedir, 'minimum_control_energy', file))
    df_energy[i] = E[indices]

# %% Fig 2a: brain maps
# add principal gradient to load_average_bms
try:
    load_average_bms.brain_maps['pg']
except KeyError:
    pg = DataVector(data=compute_gradients.gradients[:, 0], name='pg')
    pg.rescale_unit_interval()
    load_average_bms.brain_maps['pg'] = pg

cmap = 'viridis'
figwidth = 2
figratio = 0.60
figheight = figwidth * figratio
for i, key in enumerate(load_average_bms.brain_maps):

    vtx_data, plot_min, plot_max = roi_to_vtx(load_average_bms.brain_maps[key].data+1e-5,
                                              environment.parcel_names, environment.lh_annot_file)
    vtx_data = vtx_data.astype(float)

    f = plotting.plot_surf_roi(environment.fsaverage['infl_left'], roi_map=vtx_data,
                           hemi='left', view='lateral', vmin=0, vmax=1,
                           bg_map=environment.fsaverage['sulc_left'], bg_on_data=True,
                           darkness=.5, cmap=cmap, colorbar=False)
    f.set_figwidth(figwidth)
    f.set_figheight(figheight)
    f.savefig(os.path.join(environment.figdir, 'fig-2a_{0}_lat.png'.format(key)), dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

    f = plotting.plot_surf_roi(environment.fsaverage['infl_left'], roi_map=vtx_data,
                           hemi='left', view='medial', vmin=0, vmax=1,
                           bg_map=environment.fsaverage['sulc_left'], bg_on_data=True,
                           darkness=.5, cmap=cmap, colorbar=False)
    f.set_figwidth(figwidth)
    f.set_figheight(figheight)
    f.savefig(os.path.join(environment.figdir, 'fig-2a_{0}_med.png'.format(key)), dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()


# %% Fig 2b: correlations between brain maps and gradient
df_corr = pd.DataFrame(index=load_average_bms.brain_maps.keys(), columns=load_average_bms.brain_maps.keys())
df_p = pd.DataFrame(index=load_average_bms.brain_maps.keys(), columns=load_average_bms.brain_maps.keys())

for i in load_average_bms.brain_maps:
    for j in load_average_bms.brain_maps:
        df_corr.loc[i, j] = sp.stats.pearsonr(load_average_bms.brain_maps[i].data,
                                               load_average_bms.brain_maps[j].data)[0]
        df_p.loc[i, j] = sp.stats.pearsonr(load_average_bms.brain_maps[i].data,
                                            load_average_bms.brain_maps[j].data)[1]

df_corr = df_corr.astype(float)
df_p = df_p.astype(float)

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(df_corr, dtype=bool))
n_tests = np.sum(~mask.flatten())
# mask = np.logical_or(mask, df_p > (0.05 / n_tests))

f, ax = plt.subplots(1, 1, figsize=(2.5, 2.5))
sns.heatmap(df_corr, mask=mask, ax=ax, square=True, center=0, cmap='coolwarm')
f.savefig(os.path.join(environment.figdir, 'fig-2b_corr_brainmaps.png'), dpi=150, bbox_inches='tight',
          pad_inches=0.1)
plt.close()

# %% Fig 2c: energy correlate to corresponding brain maps
# drop pg
try:
    load_average_bms.brain_maps.pop('pg')
except KeyError:
    pass

n_maps = len(load_average_bms.brain_maps)
f, ax = plt.subplots(1, n_maps, figsize=(n_maps * 2.5, 2))
plt.subplots_adjust(wspace=0.3, hspace=0)

for i, key in enumerate(load_average_bms.brain_maps):
    load_average_bms.brain_maps[key].mean_between_states(compute_gradients.grad_bins)
    # load_average_bms.brain_maps[key].mean_within_states(compute_gradients.grad_bins)
    # load_average_bms.brain_maps[key].data_mean = load_average_bms.brain_maps[key].data_mean.transpose()

    my_regplot(load_average_bms.brain_maps[key].data_mean[indices], df_energy[key], key, 'Energy ({0})'.format(key), ax[i])

f.savefig(os.path.join(environment.figdir, 'fig-2c_corr_brainmaps_energy.png'), dpi=150, bbox_inches='tight',
          pad_inches=0.1)
plt.close()

# %% Fig 2d: correlations between weighted and unweighted energies
df_corr = pd.DataFrame(index=my_list, columns=my_list)
df_p = pd.DataFrame(index=my_list, columns=my_list)

for i in my_list:
    for j in my_list:
        df_corr.loc[i, j] = sp.stats.pearsonr(df_energy[i], df_energy[j])[0]
        df_p.loc[i, j] = sp.stats.pearsonr(df_energy[i], df_energy[j])[1]
df_corr = df_corr.astype(float)
df_p = df_p.astype(float)

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(df_corr, dtype=bool))
n_tests = np.sum(~mask.flatten())
# mask = np.logical_or(mask, df_p > (0.05 / n_tests))

f, ax = plt.subplots(1, 1, figsize=(2.5, 2.5))
sns.heatmap(df_corr, mask=mask, ax=ax, square=True, center=0, cmap='coolwarm')
f.savefig(os.path.join(environment.figdir, 'fig-2d_corr_energy.png'), dpi=150, bbox_inches='tight',
          pad_inches=0.1)
plt.close()
