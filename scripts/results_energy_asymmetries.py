import sys, os, platform
from pfactor_gradients.pnc import Environment, Subject
from pfactor_gradients.routines import LoadSC, LoadCT, LoadSA, LoadAverageSC, LoadAverageBrainMaps
from pfactor_gradients.pipelines import ComputeGradients, ComputeMinimumControlEnergy
from pfactor_gradients.utils import rank_int, get_fdr_p, fit_hyperplane, helper_null_hyperplane, helper_null_mean,\
get_states_from_gradient, get_null_p
from pfactor_gradients.plotting import my_reg_plot, my_null_plot, my_distpair_plot
from pfactor_gradients.imaging_derivs import DataVector, DataMatrix
from pfactor_gradients.hcp import BrainMapLoader
import numpy as np
import pandas as pd
import scipy as sp
from tqdm import tqdm

# %% Plotting
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
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

# filter subjects
filters = {'healthExcludev2': 0, 'psychoactiveMedPsychv2': 0,
           't1Exclude': 0, 'fsFinalExclude': 0,
           'b0ProtocolValidationStatus': 1, 'dti64ProtocolValidationStatus': 1, 'dti64Exclude': 0}
           # 'restProtocolValidationStatus': 1, 'restExclude': 0} # need to add these filters in if doing funcg1 below
environment.load_metadata(filters)

# %% get states
which_grad = 'histg2'

if which_grad == 'histg2':
    if computer == 'macbook':
        bbw_dir = '/Volumes/T7/research_data/BigBrainWarp/spaces/fsaverage/'
    elif computer == 'cbica':
        bbw_dir = '/cbica/home/parkesl/research_data/BigBrainWarp/spaces/fsaverage/'

    if parc == 'schaefer':
        gradient = np.loadtxt(os.path.join(bbw_dir, 'Hist_G2_Schaefer2018_{0}Parcels_17Networks.txt'.format(n_parcels)))
    elif parc == 'glasser':
        gradient = np.loadtxt(os.path.join(bbw_dir, 'Hist_G2_HCP-MMP1.txt'))
    gradient = gradient * -1
elif which_grad == 'funcg1':
    # compute function gradient
    compute_gradients = ComputeGradients(environment=environment, Subject=Subject)
    compute_gradients.run()
    gradient = compute_gradients.gradients[:, 0]

n_bins = int(n_parcels/10)
states = get_states_from_gradient(gradient=gradient, n_bins=n_bins)
n_states = len(np.unique(states))

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
    'sa': LoadSA(environment=environment, Subject=Subject)
}

load_average_bms = LoadAverageBrainMaps(loaders_dict=loaders_dict)
load_average_bms.run(return_descending=False)

# %% get control energy
file_prefix = 'average_adj_n-{0}_s-{1}_{2}_'.format(load_average_sc.load_sc.df.shape[0], spars_thresh, which_grad)
n_subsamples = 0
B_list = ['wb',] + list(load_average_bms.brain_maps.keys())
E = dict.fromkeys(B_list)

for B in B_list:
    if B == 'wb':
        nct_pipeline = ComputeMinimumControlEnergy(environment=environment, A=load_average_sc.A,
                                                   states=states, n_subsamples=n_subsamples,
                                                   control='minimum_fast', T=1, B='wb', file_prefix=file_prefix,
                                                   force_rerun=False, save_outputs=True, verbose=True)
        nct_pipeline.run()
    else:
        nct_pipeline = ComputeMinimumControlEnergy(environment=environment, A=A,
                                                   states=states, n_subsamples=n_subsamples,
                                                   control='minimum_fast', T=1, B=load_average_bms.brain_maps[B],
                                                   file_prefix=file_prefix,
                                                   force_rerun=False, save_outputs=True, verbose=True)
        nct_pipeline.run()

    E[B] = nct_pipeline.E

# %% plots

# %% data for plotting
# for B in ['wb', 'ct', 'sa']:
B = 'wb'
print(B)
e = rank_int(E[B]) # normalized energy matrix
ed = e.transpose() - e # energy asymmetry matrix
# save out mean ed for use in other scripts
np.save(os.path.join(environment.pipelinedir, 'ed_{0}_{1}.npy'.format(which_grad, B)), ed)

try:
    n_perms = 10000
    network_null = 'mni-wsp'
    # file = 'average_adj_n-{0}_s-{1}_null-{2}_ns-{3}-0_c-minimum_fast_T-1_B-{4}_E.npy'.format(n_subs, spars_thresh,
    #                                                                                              network_null, n_states, B)
    # e_network_null = np.load(os.path.join(environment.pipelinedir, 'minimum_control_energy', file))
    e_network_null = np.zeros((n_states, n_states, n_perms))
    for i in tqdm(np.arange(n_perms)):
        file = 'average_adj_n-{0}_s-{1}_{2}_null-{3}-{4}_ns-{5}-0_c-minimum_fast_T-1_B-{6}_E.npy'.format(n_subs,
                                                                                                         spars_thresh,
                                                                                                         which_grad,
                                                                                                         network_null,
                                                                                                         i, n_states, B)
        e_network_null[:, :, i] = np.load(os.path.join(environment.pipelinedir, 'minimum_control_energy', file))

        # normalize
        e_network_null[:, :, i] = rank_int(e_network_null[:, :, i])

    if B != 'wb':
        # file = 'average_adj_n-{0}_s-{1}_ns-{2}-0_c-minimum_fast_T-1_B-{3}-spin_E.npy'.format(n_subs, spars_thresh, n_states, B)
        # e_spin_null = np.load(os.path.join(environment.pipelinedir, 'minimum_control_energy', file))
        e_spin_null = np.zeros((n_states, n_states, n_perms))
        for i in tqdm(np.arange(n_perms)):
            file = 'average_adj_n-{0}_s-{1}_{2}_ns-{3}-0_c-minimum_fast_T-1_B-{4}-spin-{5}_E.npy'.format(n_subs,
                                                                                                         spars_thresh,
                                                                                                         which_grad,
                                                                                                         n_states, B, i)
            e_spin_null[:, :, i] = np.load(os.path.join(environment.pipelinedir, 'minimum_control_energy', file))

            # normalize nulls
            e_spin_null[:, :, i] = rank_int(e_spin_null[:, :, i])

except FileNotFoundError:
    print('Requisite files not found...')
    del e_network_null

# %% 1) energy dists: top-down vs bottom-up
f, ax = plt.subplots(1, 1, figsize=(figsize, figsize))
df_plot = pd.DataFrame(data=np.vstack((e[indices_upper], e[indices_lower])).transpose(),
                       columns=['bottom-up', 'top-down'])
my_distpair_plot(df=df_plot, ylabel='energy (z-score)', ax=ax)
f.savefig(os.path.join(environment.figdir, 'e_{0}'.format(B)), dpi=600, bbox_inches='tight',
          pad_inches=0.01)
plt.close()

# 2) energy asymmetry matrix
plot_mask = np.zeros((n_states, n_states))
plot_mask[indices_upper] = 1
plot_mask[np.eye(n_states) == 1] = 1
plot_mask = plot_mask.astype(bool)

f, ax = plt.subplots(1, 1, figsize=(figsize*1.2, figsize*1.2))
# sns.heatmap(rank_int(ed), mask=plot_mask, center=0, vmin=-2, vmax=2,
sns.heatmap(ed, mask=plot_mask, center=0, vmin=np.floor(np.min(ed)), vmax=np.ceil(np.max(ed)),
                        square=True, cmap='coolwarm', ax=ax, cbar_kws={"shrink": 0.80})
ax.set_ylabel("initial states", labelpad=-1)
ax.set_xlabel("target states", labelpad=-1)
ax.set_yticklabels('')
ax.set_xticklabels('')
ax.tick_params(pad=-2.5)
f.savefig(os.path.join(environment.figdir, 'e_asym_{0}'.format(B)), dpi=600, bbox_inches='tight', pad_inches=0.01)
plt.close()

# %% 2.1) energy asymmetry distance

# get hierarchy distance between states
states_distance = np.zeros((n_states, n_states))
for i in np.arange(n_states):
    for j in np.arange(n_states):
        states_distance[i, j] = gradient[states == i].mean() - gradient[states == j].mean()
states_distance = DataMatrix(data=states_distance)

# get mni distance between states
states_distance_mni = sp.spatial.distance.squareform(sp.spatial.distance.pdist(environment.centroids.values))
states_distance_mni[np.eye(states_distance_mni.shape[0]) == 1] = np.nan
states_distance_mni = DataMatrix(data=states_distance_mni)
states_distance_mni.mean_over_clusters(states)

# regress mni distance out of energy asymmetry
ed_matrix = DataMatrix(data=ed)
mask = np.zeros((n_states, n_states)).astype(bool)
mask[indices_lower] = True
ed_matrix.regress_nuisance(c=states_distance_mni.data_clusters, mask=mask)

# plot
f, ax = plt.subplots(1, 1, figsize=(figsize, figsize))
my_reg_plot(states_distance.data[indices_lower], np.abs(ed_matrix.data_resid[indices_lower]),
                        'hierarchy distance', 'energy asymmetry\n(abs.)', ax, annotate='spearman')
f.savefig(os.path.join(environment.figdir, 'corr(distance,e_asym_{0})'.format(B)), dpi=600, bbox_inches='tight',
          pad_inches=0.01)
plt.close()

# plot null
try:
    r_null = np.zeros(n_perms)

    for i in np.arange(n_perms):
        ed_null = DataMatrix(data=e_network_null[:, :, i].transpose() - e_network_null[:, :, i])
        ed_null.regress_nuisance(c=states_distance_mni.data_clusters, mask=mask)
        r_null[i] = sp.stats.spearmanr(states_distance.data[indices_lower], np.abs(ed_null.data_resid[indices_lower]))[0]

    # get p val
    observed = sp.stats.spearmanr(states_distance.data[indices_lower], np.abs(ed_matrix.data_resid[indices_lower]))[0]
    p_val = get_null_p(observed, r_null, abs=True)

    f, ax = plt.subplots(1, 1, figsize=(figsize, figsize))
    my_null_plot(observed=observed, null=r_null, p_val=p_val, xlabel='distance corr.\n(null network)', ax=ax)
    f.savefig(os.path.join(environment.figdir, 'corr(distance,e_asym_{0})_null'.format(B)), dpi=600,
              bbox_inches='tight', pad_inches=0.01)
    plt.close()
except NameError:
    print('Requisite variables not found...')

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

f.savefig(os.path.join(environment.figdir, 'e_asym_hyperplane_{0}'.format(B)), dpi=600,
          bbox_inches='tight', pad_inches=0.2)
plt.close()

# %% 4) energy asymmetry hyperplane null
try:
    asymm_nulls, observed, p_vals = helper_null_hyperplane(e, e_network_null, indices_lower)
    print(np.mean(asymm_nulls, axis=0))
    print(observed)
    print(p_vals)

    i=0
    f, ax = plt.subplots(1, 1, figsize=(figsize, figsize))
    my_null_plot(observed=observed[i], null=asymm_nulls[:, i], p_val=p_vals[i], xlabel='$R^2$\n(null network)', ax=ax)
    f.savefig(os.path.join(environment.figdir, 'e_asym_hyperplane_network_null_{0}'.format(B)), dpi=600,
              bbox_inches='tight', pad_inches=0.01)
    plt.close()
except NameError:
    print('Requisite variables not found...')

# %% 5) effective connectivity

# load dcm outputs
try:
    file = 'dcm_ns-{0}_A.mat'.format(n_states)
    dcm = sp.io.loadmat(os.path.join(environment.pipelinedir, 'spdcm', file))
    ec = dcm['A']
    ec = np.abs(ec)
    ec = rank_int(ec)
    ecd = ec.transpose() - ec

    # energy asymmetry vs effective connectivity asymmetry
    f, ax = plt.subplots(1, 1, figsize=(figsize, figsize))
    my_reg_plot(x=ed[indices_lower], y=ecd[indices_lower],
               xlabel='energy (asymmetry)', ylabel='ec (asymmetry)', ax=ax, annotate='both')
    plt.subplots_adjust(wspace=.25)
    f.savefig(os.path.join(environment.figdir, 'corr(e_asym_{0},ec_asym)'.format(B)), dpi=600, bbox_inches='tight',
              pad_inches=0.01)
    plt.close()
except FileNotFoundError:
    print('Requisite files not found...')

# %% note: the following section only runs for weighted control
if B != 'wb':
    # %% 6) correlations between control set weights and energy
    load_average_bms.brain_maps[B].mean_between_states(states) # across all regions for state pairs
    f, ax = plt.subplots(1, 1, figsize=(figsize, figsize))
    my_reg_plot(load_average_bms.brain_maps[B].data_mean[indices], E[B][indices],
               '{0}\n (averaged within state pairs)'.format(B.upper()), '{0}-weighted energy'.format(B.upper()), ax)
    f.savefig(os.path.join(environment.figdir, 'corr({0},energy_{0})'.format(B)), dpi=600, bbox_inches='tight',
              pad_inches=0.01)
    plt.close()

    load_average_bms.brain_maps[B].mean_within_states(states) # across regions within target states
    f, ax = plt.subplots(1, 1, figsize=(figsize, figsize))
    my_reg_plot(load_average_bms.brain_maps[B].data_mean[indices], E[B][indices],
               '{0}\n (averaged within target states)'.format(B.upper()), '{0}-weighted energy'.format(B.upper()), ax)
    f.savefig(os.path.join(environment.figdir, 'corr({0}_target,energy_{0})'.format(B)), dpi=600, bbox_inches='tight',
              pad_inches=0.01)
    plt.close()

    load_average_bms.brain_maps[B].data_mean = load_average_bms.brain_maps[B].data_mean.transpose() # across regions within initial states
    f, ax = plt.subplots(1, 1, figsize=(figsize, figsize))
    my_reg_plot(load_average_bms.brain_maps[B].data_mean[indices], E[B][indices],
               '{0}\n (averaged within initial states)'.format(B.upper()), '{0}-weighted energy'.format(B.upper()), ax)
    f.savefig(os.path.join(environment.figdir, 'corr({0}_initial,energy_{0})'.format(B)), dpi=600, bbox_inches='tight',
              pad_inches=0.01)
    plt.close()

    # %% 7) spin test null
    try:
        asymm_null, observed, p_val = helper_null_mean(e, e_spin_null, indices_lower)
        f, ax = plt.subplots(1, 1, figsize=(figsize, 0.75))
        my_null_plot(observed=observed, null=asymm_null, p_val=p_val, xlabel='mean asym. (spin-test)', ax=ax)
        f.savefig(os.path.join(environment.figdir, 'e_asym_mean_spin_null_{0}'.format(B)), dpi=600,
                  bbox_inches='tight', pad_inches=0.01)
        plt.close()
    except NameError:
        print('Requisite variables not found...')

    # %% 8) null network model
    try:
        asymm_null, observed, p_val = helper_null_mean(e, e_network_null, indices_lower)
        f, ax = plt.subplots(1, 1, figsize=(figsize, 0.75))
        my_null_plot(observed=observed, null=asymm_null, p_val=p_val, xlabel='mean asym. (null network)', ax=ax)
        f.savefig(os.path.join(environment.figdir, 'e_asym_mean_network_null_{0}'.format(B)), dpi=600,
                  bbox_inches='tight', pad_inches=0.01)
        plt.close()
    except NameError:
        print('Requisite variables not found...')
