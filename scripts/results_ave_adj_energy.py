# %% import
import sys, os, platform
from pfactor_gradients.imaging_derivs import DataMatrix, DataVector
from pfactor_gradients.pipelines import ComputeMinimumControlEnergy
from pfactor_gradients.plotting import my_reg_plot, my_distpair_plot, my_null_plot
from pfactor_gradients.energy import expand_states, matrix_normalization
from pfactor_gradients.utils import rank_int, get_null_p, get_exact_p, get_fdr_p

import numpy as np
import pandas as pd
import scipy as sp
from scipy.linalg import svd
from tqdm import tqdm

# %% import workspace
from setup_workspace_ave_adj import *

# %% plotting
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
from pfactor_gradients.plotting import set_plotting_params
set_plotting_params(format='png')
figsize = 1.5

# %% brain maps
A_tmp = DataMatrix(data=A)
A_tmp.get_strength()

for key in load_average_bms.brain_maps:
    nan_mask = np.isnan(load_average_bms.brain_maps[key].data)

    # load_average_bms.brain_maps[key].regress_nuisance(state_brain_map)
    # load_average_bms.brain_maps[key].data = load_average_bms.brain_maps[key].data_resid.copy()
    # load_average_bms.brain_maps[key].rankdata()
    # load_average_bms.brain_maps[key].rescale_unit_interval()

    print('state_brain_map vs. {0}'.format(key), sp.stats.pearsonr(state_brain_map[~nan_mask],
                                                                   load_average_bms.brain_maps[key].data[~nan_mask]))
    print('strength vs. {0}'.format(key), sp.stats.spearmanr(A_tmp.S[~nan_mask],
                                                             load_average_bms.brain_maps[key].data[~nan_mask]))

    # plot brain map
    load_average_bms.brain_maps[key].brain_surface_plot(environment)

# plot state brain map
DataVector(data=state_brain_map, name='state_brain_map').brain_surface_plot(environment)
DataVector(data=states == 0, name='state_0').brain_surface_plot(environment)
DataVector(data=states == int(n_states/2), name='state_{0}'.format(int(n_states/2))).brain_surface_plot(environment)
DataVector(data=states == n_states-1, name='state_{0}'.format(int(n_states-1))).brain_surface_plot(environment)

# %% get control energy
file_prefix = 'average_adj_n-{0}_cthr-{1}_smap-{2}_'.format(load_average_sc.load_sc.df.shape[0],
                                                            consist_thresh, which_brain_map)

B_dict = dict()

B = DataMatrix(data=np.eye(n_parcels), name='identity')
B_dict[B.name] = B

E = dict.fromkeys(B_dict)

c = 1
T = 1

for B in B_dict:
    nct_pipeline = ComputeMinimumControlEnergy(environment=environment, A=A, states=states, B=B_dict[B],
                                               control='minimum_fast', c=c, T=T,
                                               file_prefix=file_prefix,
                                               force_rerun=False, save_outputs=True, verbose=True)
    nct_pipeline.run()

    E[B] = nct_pipeline.E


# optimized B weights
nct_pipeline = ComputeMinimumControlEnergy(environment=environment, A=A, states=states, B=B_dict['identity'],
                                           control='minimum_fast', c=c, T=T,
                                           file_prefix=file_prefix,
                                           force_rerun=False, save_outputs=True, verbose=True)
n = 2
ds = 0.1
nct_pipeline.run_with_optimized_b(n=n, ds=ds)

E['E_opt'] = nct_pipeline.E_opt[:, 1].reshape(n_states, n_states)
B_opt = nct_pipeline.B_opt[:, :, 1]

# %% data for plotting
norm_energy = True
# for B in ['identity', 'E_opt']:
for B in ['identity', ]:
    # B = 'identity'
    # B = 'E_opt'
    print(B)
    e = E[B].copy()

    if norm_energy:
        e = rank_int(e) # normalized energy matrix

    ed = e - e.transpose() # energy asymmetry matrix
    print(np.all(np.round(np.abs(ed.flatten()), 4) == np.round(np.abs(ed.transpose().flatten()), 4)))

    # save out mean ed for use in other scripts
    np.save(os.path.join(environment.pipelinedir, 'e_{0}_{1}.npy'.format(which_brain_map, B)), e)
    np.save(os.path.join(environment.pipelinedir, 'ed_{0}_{1}.npy'.format(which_brain_map, B)), ed)

    try:
        n_perms = 10000
        network_null = 'mni-wwp'
        e_network_null = np.zeros((n_states, n_states, n_perms))

        for i in tqdm(np.arange(n_perms)):
            file = 'average_adj_n-{0}_cthr-{1}_smap-{2}_null-{3}-{4}_ns-{5}_ctrl-minimum_fast_c-{6}_T-{7}_B-{8}_E.npy' \
                .format(n_subs,
                        consist_thresh,
                        which_brain_map,
                        network_null,
                        i, n_states, c, T, B)
            e_network_null[:, :, i] = np.load(os.path.join(environment.pipelinedir, 'minimum_control_energy_new', file))

            # normalize
            if norm_energy:
                e_network_null[:, :, i] = rank_int(e_network_null[:, :, i])

    except FileNotFoundError:
        print('Requisite files not found...')
        del e_network_null

    # %% 1) energy matrix
    plot_mask = np.eye(n_states)
    plot_mask = plot_mask.astype(bool)

    f, ax = plt.subplots(1, 1, figsize=(figsize*1.2, figsize*1.2))
    sns.heatmap(e, mask=plot_mask, center=0, vmin=np.floor(np.min(e[~plot_mask])), vmax=np.ceil(np.max(e)),
                            square=True, cmap='coolwarm', ax=ax, cbar_kws={"shrink": 0.80})
    ax.set_ylabel("initial states", labelpad=-1)
    ax.set_xlabel("target states", labelpad=-1)
    ax.set_yticklabels('')
    ax.set_xticklabels('')
    ax.tick_params(pad=-2.5)
    f.savefig(os.path.join(environment.figdir, 'e_{0}'.format(B)), dpi=600, bbox_inches='tight', pad_inches=0.01)
    plt.close()

    # %% 2) energy asymmetry

    # top-down vs bottom-up
    f, ax = plt.subplots(1, 1, figsize=(figsize, figsize))
    df_plot = pd.DataFrame(data=np.vstack((e[indices_upper], e[indices_lower])).transpose(),
                           columns=['bottom-up', 'top-down'])
    my_distpair_plot(df=df_plot, ylabel='energy (z-score)', ax=ax)
    f.savefig(os.path.join(environment.figdir, 'e_asym_{0}_dists'.format(B)), dpi=600, bbox_inches='tight',
              pad_inches=0.01)
    plt.close()

    # energy asymmetry matrix
    plot_mask = np.zeros((n_states, n_states))
    plot_mask[indices_lower] = 1
    plot_mask[np.eye(n_states) == 1] = 1
    plot_mask = plot_mask.astype(bool)

    f, ax = plt.subplots(1, 1, figsize=(figsize*1.2, figsize*1.2))
    sns.heatmap(ed, mask=plot_mask, center=0, vmin=np.floor(np.min(ed)), vmax=np.ceil(np.max(ed)),
                            square=True, cmap='coolwarm', ax=ax, cbar_kws={"shrink": 0.80})
    ax.set_ylabel("initial states", labelpad=-1)
    ax.set_xlabel("target states", labelpad=-1)
    ax.set_yticklabels('')
    ax.set_xticklabels('')
    ax.tick_params(pad=-2.5)
    f.savefig(os.path.join(environment.figdir, 'e_asym_{0}_matrix'.format(B)), dpi=600, bbox_inches='tight', pad_inches=0.01)
    plt.close()

    # %% 3) energy asymmetry distance corr

    # get hierarchy distance between states
    states_distance = np.zeros((n_states, n_states))
    for i in np.arange(n_states):
        for j in np.arange(n_states):
            states_distance[i, j] = state_brain_map[states == i].mean() - state_brain_map[states == j].mean()
    states_distance = np.abs(states_distance)
    states_distance = DataMatrix(data=states_distance)

    # get mni distance between states
    states_distance_mni = sp.spatial.distance.squareform(sp.spatial.distance.pdist(environment.centroids.values))
    states_distance_mni[np.eye(states_distance_mni.shape[0]) == 1] = np.nan
    states_distance_mni = DataMatrix(data=states_distance_mni)
    states_distance_mni.mean_over_clusters(states)

    # regress mni distance out of energy asymmetry
    ed_matrix = DataMatrix(data=ed)
    mask = np.zeros((n_states, n_states)).astype(bool)
    mask[indices_upper] = True
    ed_matrix.regress_nuisance(c=states_distance_mni.data_clusters, mask=mask)

    # plot distance asymm
    f, ax = plt.subplots(1, 1, figsize=(figsize, figsize))
    my_reg_plot(states_distance.data[indices_upper], ed_matrix.data[indices_upper],
                            'hierarchy distance', 'energy asymmetry', ax, annotate='spearman')
    f.savefig(os.path.join(environment.figdir, 'corr(distance,e_asym_{0})'.format(B)), dpi=600, bbox_inches='tight',
              pad_inches=0.01)
    plt.close()

    # plot distance asymm (absolute)
    f, ax = plt.subplots(1, 1, figsize=(figsize, figsize))
    my_reg_plot(states_distance.data[indices_upper], np.abs(ed_matrix.data[indices_upper]),
                            'hierarchy distance', 'energy asymmetry\n(abs.)', ax, annotate='spearman')
    f.savefig(os.path.join(environment.figdir, 'corr(distance,abs_e_asym_{0})'.format(B)), dpi=600, bbox_inches='tight',
              pad_inches=0.01)
    plt.close()

    # plot null
    try:
        r_null = np.zeros(n_perms)
        r_null_abs = np.zeros(n_perms)

        for i in np.arange(n_perms):
            ed_null = DataMatrix(data=e_network_null[:, :, i] - e_network_null[:, :, i].transpose())
            ed_null.regress_nuisance(c=states_distance_mni.data_clusters, mask=mask)

            r_null[i] = sp.stats.spearmanr(states_distance.data[indices_upper], ed_null.data[indices_upper])[0]
            r_null_abs[i] = sp.stats.spearmanr(states_distance.data[indices_upper], np.abs(ed_null.data[indices_upper]))[0]

        # plot distance asymm null
        observed = sp.stats.spearmanr(states_distance.data[indices_upper], ed_matrix.data[indices_upper])[0]
        p_val = get_null_p(observed, r_null, abs=True)
        f, ax = plt.subplots(1, 1, figsize=(figsize, figsize))
        my_null_plot(observed=observed, null=r_null, p_val=p_val, xlabel='distance corr.\n(null network)', ax=ax)
        f.savefig(os.path.join(environment.figdir, 'corr(distance,abs_e_asym_{0})_null'.format(B)), dpi=600,
                  bbox_inches='tight', pad_inches=0.01)
        plt.close()

        # plot distance asymm null (absolute)
        observed = sp.stats.spearmanr(states_distance.data[indices_upper], np.abs(ed_matrix.data[indices_upper]))[0]
        p_val = get_null_p(observed, r_null_abs, abs=True)
        f, ax = plt.subplots(1, 1, figsize=(figsize, figsize))
        my_null_plot(observed=observed, null=r_null_abs, p_val=p_val, xlabel='distance corr.\n(null network)', ax=ax)
        f.savefig(os.path.join(environment.figdir, 'corr(distance,e_asym_{0})_null'.format(B)), dpi=600,
                  bbox_inches='tight', pad_inches=0.01)
        plt.close()
    except NameError:
        print('Requisite variables not found...')

    # %% 4) effective connectivity

    # load dcm outputs
    try:
        file = 'dcm_ns-{0}_A.mat'.format(n_states)
        dcm = sp.io.loadmat(os.path.join(environment.pipelinedir, 'spdcm', file))
        ec = dcm['A']
        ec = np.abs(ec)
        ec = rank_int(ec)
        ecd = ec - ec.transpose()

        # effective connectivity matrix
        f, ax = plt.subplots(1, 2, figsize=(figsize*2.4, figsize*1.2))
        sns.heatmap(ec, center=0, vmin=np.floor(np.min(ec)), vmax=np.ceil(np.max(ec)),
                    square=True, cmap='coolwarm', ax=ax[0], cbar_kws={"shrink": 0.60})
        sns.heatmap(ecd, center=0, vmin=np.floor(np.min(ecd)), vmax=np.ceil(np.max(ecd)),
                    square=True, cmap='coolwarm', ax=ax[1], cbar_kws={"shrink": 0.60})
        for i in [0, 1]:
            ax[i].set_ylabel("initial states", labelpad=-1)
            ax[i].set_xlabel("target states", labelpad=-1)
            ax[i].set_yticklabels('')
            ax[i].set_xticklabels('')
            ax[i].tick_params(pad=-2.5)
        f.savefig(os.path.join(environment.figdir, 'ec_matrix'), dpi=600, bbox_inches='tight', pad_inches=0.01)
        plt.close()

        # energy asymmetry vs effective connectivity asymmetry
        f, ax = plt.subplots(1, 1, figsize=(figsize, figsize))
        my_reg_plot(x=ed[indices_upper], y=ecd[indices_upper],
                    xlabel='energy (delta)', ylabel='effective connectivity (delta)',
                    ax=ax, annotate='both')
        plt.subplots_adjust(wspace=.25)
        f.savefig(os.path.join(environment.figdir, 'ed_ecd_{0}'.format(B)), dpi=600,
                  bbox_inches='tight', pad_inches=0.1)
        plt.close()
    except FileNotFoundError:
        print('Requisite files not found...')

    # %% 5) fMRI delta
    timescales_delta = np.zeros((n_states, n_states))
    for i in np.arange(n_states):
        for j in np.arange(n_states):
            timescales_delta[i, j] = np.nanmean(load_average_bms.brain_maps['tau'].data[states == i]) - \
                                     np.nanmean(load_average_bms.brain_maps['tau'].data[states == j])
    # sign of this timescales_delta matrix is currently unintuitive.
    #   if state_i = 0.3 and state_j = 0.5, then 0.3-0.5=-0.2.
    #   likewise, if state_i = 0.5 and state_j = 0.3, then 0.5-0.3=0.2.
    # thus, an increase in rlfp over states is encoded by a negative number and a decrease is encoded by a positive
    # number. Not good! sign flip for intuition
    timescales_delta = timescales_delta * -1
    # now, negative sign represent bold power decreasing over states and positive sign represent bold power increasing over states.

    # fmri matrix
    f, ax = plt.subplots(1, 1, figsize=(figsize*1.2, figsize*1.2))
    sns.heatmap(timescales_delta, center=0, vmin=np.floor(np.min(timescales_delta)), vmax=np.ceil(np.max(timescales_delta)),
                square=True, cmap='coolwarm', ax=ax, cbar_kws={"shrink": 0.60})
    ax.set_ylabel("initial states", labelpad=-1)
    ax.set_xlabel("target states", labelpad=-1)
    ax.set_yticklabels('')
    ax.set_xticklabels('')
    ax.tick_params(pad=-2.5)
    f.savefig(os.path.join(environment.figdir, 'timescales_delta_matrix'), dpi=600, bbox_inches='tight', pad_inches=0.01)
    plt.close()

    # energy asymmetry vs RLFP delta
    f, ax = plt.subplots(1, 1, figsize=(figsize, figsize))
    my_reg_plot(x=timescales_delta[indices_upper], y=ed[indices_upper],
                xlabel='timescales (delta)', ylabel='energy asymmetry',
                ax=ax, annotate='both')
    plt.subplots_adjust(wspace=.25)
    f.savefig(os.path.join(environment.figdir, 'ed_timescales_{0}'.format(B)), dpi=600,
              bbox_inches='tight', pad_inches=0.1)
    plt.close()
