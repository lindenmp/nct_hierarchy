import sys, os, platform
from pfactor_gradients.pnc import Environment, Subject
from pfactor_gradients.routines import LoadSC, LoadCT, LoadCBF, LoadAverageSC, LoadAverageBrainMaps
from pfactor_gradients.pipelines import ComputeGradients, ComputeMinimumControlEnergy
from pfactor_gradients.utils import rank_int, get_fdr_p, fit_hyperplane, helper_null_hyperplane, helper_null_mean,\
    get_null_p
from pfactor_gradients.plotting import my_reg_plot, my_null_plot, my_distpair_plot, roi_to_vtx
from pfactor_gradients.imaging_derivs import DataVector
from pfactor_gradients.hcp import BrainMapLoader
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

# %% data for plotting
# for B in ['wb', 'ct', 'cbf']:
B = 'ct'
print(B)
e = rank_int(E[B]) # normalized energy matrix
ed = e.transpose() - e # energy asymmetry matrix
# save out mean ed for use in other scripts
np.save(os.path.join(environment.pipelinedir, 'ed_{0}.npy'.format(B)), ed)

network_null = 'wwp'
file = 'average_adj_n-{0}_s-{1}_null-mni-{2}_ns-{3}-0_c-minimum_fast_T-1_B-{4}_E.npy'.format(n_subs, spars_thresh,
                                                                                             network_null, n_states, B)
e_network_null = np.load(os.path.join(environment.pipelinedir, 'minimum_control_energy', file))
# normalize nulls
for i in np.arange(e_network_null.shape[2]):
    e_network_null[:, :, i] = rank_int(e_network_null[:, :, i])

if B != 'wb':
    file = 'average_adj_n-{0}_s-{1}_ns-{2}-0_c-minimum_fast_T-1_B-{3}-spin_E.npy'.format(n_subs, spars_thresh, n_states, B)
    e_spin_null = np.load(os.path.join(environment.pipelinedir, 'minimum_control_energy', file))

    # normalize nulls
    for i in np.arange(e_spin_null.shape[2]):
        e_spin_null[:, :, i] = rank_int(e_spin_null[:, :, i])

# %% 1) energy dists: top-down vs bottom-up
f, ax = plt.subplots(1, 1, figsize=(figsize, figsize))
df_plot = pd.DataFrame(data=np.vstack((e[indices_upper], e[indices_lower])).transpose(),
                       columns=['bottom-up', 'top-down'])
my_distpair_plot(df=df_plot, ylabel='energy (z-score)', ax=ax)
f.savefig(os.path.join(environment.figdir, 'e_{0}'.format(B)), dpi=300, bbox_inches='tight',
          pad_inches=0.01)
plt.close()

# 2) energy asymmetry matrix
plot_mask = np.zeros((n_states, n_states))
plot_mask[indices_upper] = 1
plot_mask = plot_mask.astype(bool)

f, ax = plt.subplots(1, 1, figsize=(figsize, figsize))
sns.heatmap(ed, mask=plot_mask, center=0, square=True, cmap='coolwarm', ax=ax, cbar_kws={"shrink": 0.80})
ax.set_ylabel("initial states", labelpad=-1)
ax.set_xlabel("target states", labelpad=-1)
ax.set_yticklabels('')
ax.set_xticklabels('')
ax.tick_params(pad=-2.5)
f.savefig(os.path.join(environment.figdir, 'e_asym_{0}'.format(B)), dpi=300, bbox_inches='tight', pad_inches=0.01)
plt.close()

# %% 3) energy asymmetry hyperplane
plot_data = np.concatenate((indices_lower[0].reshape(-1, 1),
                            indices_lower[1].reshape(-1, 1),
                            ed[indices_lower].reshape(-1, 1)), axis=1)
plot_data = (plot_data - plot_data.mean(axis=0)) / plot_data.std(axis=0)
# fit hyperplane
X, Y, Z, c, r2, mse, rmse = fit_hyperplane(plot_data)

f = plt.figure(figsize=(figsize, figsize))
ax = Axes3D(f)
cmap = ListedColormap(sns.color_palette("coolwarm", 256, ).as_hex())
sc = ax.scatter(plot_data[:, 0], plot_data[:, 1], plot_data[:, 2], marker='o', alpha=1, s=10, c=plot_data[:, 2],
                cmap=cmap, vmin=-3, vmax=2)
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.2)
ax.tick_params(pad=-2.5)
ax.set_xticklabels('')
ax.set_yticklabels('')
ax.set_zticklabels('')
ax.set_xlabel('initial states\n$\\beta$ = {:.2f}'.format(c[0]), labelpad=-10)
ax.set_ylabel('target states\n$\\beta$ = {:.2f}'.format(c[1]), labelpad=-10)
ax.set_zlabel('energy asymmetry')
ax.view_init(20, 45)
# plt.legend(*sc.legend_elements(), bbox_to_anchor=(1.05, 1), loc=2)
textstr = '$R^2$ = {:.0f}%'.format(r2*100)
ax.text2D(0.5, 0.95, textstr, transform=ax.transAxes,
        horizontalalignment='center', verticalalignment='center', rotation='horizontal')

f.savefig(os.path.join(environment.figdir, 'e_asym_hyperplane_{0}'.format(B)), dpi=300,
          bbox_inches='tight', pad_inches=0.2)
plt.close()

# %% 4) energy asymmetry hyperplane null
asymm_nulls, observed, p_vals = helper_null_hyperplane(e, e_network_null, indices_lower)
print(np.mean(asymm_nulls, axis=0))
print(observed)
print(p_vals)

i=0
f, ax = plt.subplots(1, 1, figsize=(figsize, figsize))
my_null_plot(observed=observed[i], null=asymm_nulls[:, i], p_val=p_vals[i], xlabel='$R^2$\n(null network)', ax=ax)
f.savefig(os.path.join(environment.figdir, 'e_asym_hyperplane_network_null_{0}'.format(B)), dpi=300,
          bbox_inches='tight', pad_inches=0.01)
plt.close()

# %% 5) effective connectivity

# load dcm outputs
file = 'dcm_ns-{0}_A.mat'.format(n_states)
dcm = sp.io.loadmat(os.path.join(environment.pipelinedir, 'spdcm', file))
ec = dcm['A']
ec = np.abs(ec)
ec = rank_int(ec)
ecd = ec.transpose() - ec

# energy asymmetry vs effective connectivity asymmetry
f, ax = plt.subplots(1, 1, figsize=(figsize, figsize))
my_reg_plot(x=ed[indices_lower], y=ecd[indices_lower],
           xlabel='energy (asymmetry)', ylabel='ec (asymmetry)', ax=ax)
plt.subplots_adjust(wspace=.25)
f.savefig(os.path.join(environment.figdir, 'corr(e_asym_{0},ec_asym)'.format(B)), dpi=300, bbox_inches='tight',
          pad_inches=0.01)
plt.close()

# %% note: the following section only runs for weighted control
if B != 'wb':
    # %% 6) correlations between control set weights and energy
    load_average_bms.brain_maps[B].mean_between_states(compute_gradients.grad_bins) # across all regions for state pairs
    # load_average_bms.brain_maps[B].mean_within_states(compute_gradients.grad_bins) # across regions within target states
    # load_average_bms.brain_maps[B].data_mean = load_average_bms.brain_maps[B].data_mean.transpose() # across regions within initial states

    f, ax = plt.subplots(1, 1, figsize=(figsize, figsize))
    my_reg_plot(load_average_bms.brain_maps[B].data_mean[indices], E[B][indices],
               '{0}\n (averaged within state pairs)'.format(B.upper()), '{0}-weighted energy'.format(B.upper()), ax)
    f.savefig(os.path.join(environment.figdir, 'corr({0},energy_{0})'.format(B)), dpi=300, bbox_inches='tight',
              pad_inches=0.01)
    plt.close()

    # %% 7) spin test null
    asymm_null, observed, p_val = helper_null_mean(e, e_spin_null, indices_lower)
    f, ax = plt.subplots(1, 1, figsize=(figsize, 0.75))
    my_null_plot(observed=observed, null=asymm_null, p_val=p_val, xlabel='mean asym. (spin-test)', ax=ax)
    f.savefig(os.path.join(environment.figdir, 'e_asym_mean_spin_null_{0}'.format(B)), dpi=300,
              bbox_inches='tight', pad_inches=0.01)
    plt.close()

    # %% 8) null network model
    asymm_null, observed, p_val = helper_null_mean(e, e_network_null, indices_lower)
    f, ax = plt.subplots(1, 1, figsize=(figsize, 0.75))
    my_null_plot(observed=observed, null=asymm_null, p_val=p_val, xlabel='mean asym. (null network)', ax=ax)
    f.savefig(os.path.join(environment.figdir, 'e_asym_mean_network_null_{0}'.format(B)), dpi=300,
              bbox_inches='tight', pad_inches=0.01)
    plt.close()
