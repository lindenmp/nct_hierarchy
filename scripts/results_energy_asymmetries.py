import sys, os, platform
from pfactor_gradients.pnc import Environment, Subject
from pfactor_gradients.routines import LoadSC, LoadCT, LoadCBF, LoadAverageSC, LoadAverageBrainMaps
from pfactor_gradients.pipelines import ComputeGradients, ComputeMinimumControlEnergy
from pfactor_gradients.utils import rank_int, get_fdr_p, fit_hyperplane, helper_null_hyperplane, helper_null_mean
from pfactor_gradients.plotting import my_regplot, my_nullplot, roi_to_vtx
import numpy as np
import pandas as pd
import scipy as sp

# %% Plotting
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
from nilearn import plotting

sns.set(style='whitegrid', context='paper', font_scale=1)
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
n_bins = int(n_parcels/10)
compute_gradients = ComputeGradients(environment=environment, Subject=Subject, n_bins=n_bins)
compute_gradients.run()

n_states = len(np.unique(compute_gradients.grad_bins))
mask = ~np.eye(n_states, dtype=bool)
indices = np.where(mask)
indices_upper = np.triu_indices(n_states, k=1)
indices_lower = np.tril_indices(n_states, k=-1)

# %% Load sc data
load_sc = LoadSC(environment=environment, Subject=Subject)
load_sc.run()
# refilter environment due to LoadSC excluding on disconnected nodes
environment.df = load_sc.df.copy()
n_subs = environment.df.shape[0]

if parc == 'schaefer' and n_parcels == 400:
    spars_thresh = 0.06
elif parc == 'schaefer' and n_parcels == 200:
    spars_thresh = 0.12
elif parc == 'glasser' and n_parcels == 360:
    spars_thresh = 0.07

load_average_sc = LoadAverageSC(load_sc=load_sc, spars_thresh=spars_thresh)
load_average_sc.run()
A = load_average_sc.A.copy()

# %% load mean brain maps
loaders_dict = {
    'ct': LoadCT(environment=environment, Subject=Subject),
    'cbf': LoadCBF(environment=environment, Subject=Subject)
}

load_average_bms = LoadAverageBrainMaps(loaders_dict=loaders_dict)
load_average_bms.run(return_descending=False)

# %% get control energy
file_prefix = 'average_adj_n-{0}_s-{1}_'.format(load_average_sc.load_sc.df.shape[0], spars_thresh)
n_subsamples = 0
B_list = ['wb',] + list(load_average_bms.brain_maps.keys())
E = dict.fromkeys(B_list)

for B in B_list:
    if B == 'wb':
        nct_pipeline = ComputeMinimumControlEnergy(environment=environment, A=load_average_sc.A,
                                                   states=compute_gradients.grad_bins, n_subsamples=n_subsamples,
                                                   control='minimum_fast', T=1, B='wb', file_prefix=file_prefix,
                                                   force_rerun=False, save_outputs=True, verbose=True)
        nct_pipeline.run()
    else:
        nct_pipeline = ComputeMinimumControlEnergy(environment=environment, A=A,
                                                   states=compute_gradients.grad_bins, n_subsamples=n_subsamples,
                                                   control='minimum_fast', T=1, B=load_average_bms.brain_maps[B],
                                                   file_prefix=file_prefix,
                                                   force_rerun=False, save_outputs=True, verbose=True)
        nct_pipeline.run()

    E[B] = nct_pipeline.E

# %% plots

# data for plotting
B = 'wb'
e = E[B] # energy matrix
ed = e.transpose() - e # energy asymmetry matrix

network_null = 'wwp'
file = 'average_adj_n-{0}_s-{1}_null-mni-{2}_ns-{3}-0_c-minimum_fast_T-1_B-{4}_E.npy'.format(n_subs, spars_thresh,
                                                                                             network_null, n_states, B)
e_network_null = np.load(os.path.join(environment.pipelinedir, 'minimum_control_energy', file))

if B != 'wb':
    file = 'average_adj_n-{0}_s-{1}_ns-{2}-0_c-minimum_fast_T-1_B-{3}-spin_E.npy'.format(n_subs, spars_thresh, n_states, B)
    e_spin_null = np.load(os.path.join(environment.pipelinedir, 'minimum_control_energy', file))

# %% 1) energy dists: top-down vs bottom-up
e_norm = rank_int(e)
df_plot = pd.DataFrame(data=np.vstack((e_norm[indices_upper], e_norm[indices_lower])).transpose(),
                       columns=['Bottom-up', 'Top-down'])

f, ax = plt.subplots(1, 1, figsize=(2.5, 2.5))
sns.violinplot(data=df_plot, ax=ax, inner="box", palette="pastel", cut=2, linewidth=2)
sns.despine(left=True, bottom=True)
ax.set_ylabel("Energy (z-score)")
ax.tick_params(pad=-2.5)
t, p_val = sp.stats.ttest_rel(a=e_norm[indices_upper], b=e_norm[indices_lower])
textstr = 't = {:.2f}; p = {:.2f}'.format(np.abs(t), p_val)
ax.text(0.5, ax.get_ylim()[1]+ax.get_ylim()[1]*0.06, textstr,
        horizontalalignment='center', verticalalignment='top')
ax.axhline(y=ax.get_ylim()[1]-ax.get_ylim()[1]*0.05, xmin=0.25, xmax=0.75, color='k', linewidth=1)
f.savefig(os.path.join(environment.figdir, 'e_{0}.png'.format(B)), dpi=300, bbox_inches='tight',
          pad_inches=0.1)
plt.close()

# 2) energy asymmetry matrix
plot_mask = np.zeros((n_states, n_states))
plot_mask[indices_upper] = 1
plot_mask = plot_mask.astype(bool)

f, ax = plt.subplots(1, 1, figsize=(2.5, 2.5))
sns.heatmap(rank_int(ed), mask=plot_mask, center=0, square=True, cmap='coolwarm', ax=ax, cbar_kws={"shrink": 0.80})
ax.set_ylabel("Initial states (i)")
ax.set_xlabel("Target states (j)")
ax.tick_params(pad=-2.5)
f.savefig(os.path.join(environment.figdir, 'e_asym_{0}.png'.format(B)), dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.close()

# %% 3) energy asymmetry hyperplane
ed_norm = rank_int(ed)
plot_data = np.concatenate((indices_lower[0].reshape(-1, 1),
                            indices_lower[1].reshape(-1, 1),
                            ed_norm[indices_lower].reshape(-1, 1)), axis=1)
plot_data = (plot_data - plot_data.mean(axis=0)) / plot_data.std(axis=0)
# fit hyperplane
X, Y, Z, c, r2, mse, rmse = fit_hyperplane(plot_data)

f = plt.figure(figsize=(2.5, 2.5))
ax = Axes3D(f)
cmap = ListedColormap(sns.color_palette("coolwarm", 256, ).as_hex())
sc = ax.scatter(plot_data[:, 0], plot_data[:, 1], plot_data[:, 2], marker='o', alpha=1, s=10, c=plot_data[:, 2],
                cmap=cmap, vmin=-3, vmax=2)
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.2)
ax.tick_params(pad=-2.5)
ax.set_xticklabels('')
ax.set_yticklabels('')
ax.set_zticklabels('')
ax.set_xlabel('Initial states (i) \n slope: {:.2f}'.format(c[0]), labelpad=-10)
ax.set_ylabel('Target states (j) \n slope: {:.2f}'.format(c[1]), labelpad=-10)
ax.set_zlabel('Energy asymmetry')
ax.view_init(20, 45)
# plt.legend(*sc.legend_elements(), bbox_to_anchor=(1.05, 1), loc=2)
textstr = 'r2 = {:.0f}%'.format(r2*100)
ax.text2D(0.65, 0.75, textstr, transform=ax.transAxes,
        verticalalignment='top', rotation='horizontal')

f.savefig(os.path.join(environment.figdir, 'e_asym_hyperplane_{0}.png'.format(B)), dpi=300,
          bbox_inches='tight', pad_inches=0.1)
plt.close()

# %% 4) energy asymmetry hyperplane null
asymm_nulls, observed, p_vals = helper_null_hyperplane(e, e_network_null, indices_lower)
print(np.mean(asymm_nulls, axis=0))
print(observed)
print(p_vals)

i=0
f, ax = plt.subplots(1, 1, figsize=(2.5, 2.5))
my_nullplot(observed=observed[i], null=asymm_nulls[:, i], p_val=p_vals[i], xlabel='r2 (null network)', ax=ax)
f.savefig(os.path.join(environment.figdir, 'e_asym_hyperplane_network_null_{0}.png'.format(B)), dpi=300,
          bbox_inches='tight', pad_inches=0.1)
plt.close()

# %% 5) effective connectivity

# load dcm outputs
file = 'dcm_ns-{0}_A.mat'.format(n_states)
dcm = sp.io.loadmat(os.path.join(environment.pipelinedir, 'spdcm', file))
ec = dcm['A']
ec = np.abs(ec)
ec = rank_int(ec)
ecd = ec.transpose() - ec
# ecd = rank_int(ecd)

# get energy asym
ed_norm = rank_int(ed)

# energy asymmetry vs effective connectivity asymmetry
f, ax = plt.subplots(1, 1, figsize=(2.5, 2.5))
my_regplot(x=ed_norm[indices_lower], y=ecd[indices_lower],
           xlabel='Energy (asymmetry)', ylabel='Effective connectivity (asymmetry)', ax=ax)
plt.subplots_adjust(wspace=.25)
f.savefig(os.path.join(environment.figdir, 'corr(e_asym_{0},ec_asym).png'.format(B)), dpi=300, bbox_inches='tight',
          pad_inches=0.1)
plt.close()
