import sys, os, platform
from pfactor_gradients.pnc import Environment, Subject
from pfactor_gradients.routines import LoadSC, LoadCT, LoadCBF, LoadAverageSC, LoadAverageBrainMaps
from pfactor_gradients.pipelines import ComputeGradients, ComputeMinimumControlEnergy
from pfactor_gradients.utils import rank_int, get_fdr_p, fit_hyperplane
from pfactor_gradients.plotting import my_regplot
import numpy as np
import pandas as pd
import scipy as sp

# %% Plotting
import matplotlib.pyplot as plt
import seaborn as sns

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

# %% helper funcs
def helper_null_mean(e, e_null, indices):
    n_perms = e_null.shape[2]
    # compute energy asymmetry
    ed = e.transpose() - e

    # containers
    ed_null = np.zeros(e_null.shape)
    asymm_null = np.zeros(n_perms)

    for i in np.arange(e_null.shape[2]):
        # compute null asymmetry matrix
        ed_null[:, :, i] = e_null[:, :, i].transpose() - e_null[:, :, i]
        # get mean of null asymmetry
        asymm_null[i] = np.mean(ed_null[:, :, i][indices])

    # get observed
    observed = np.mean(ed[indices])
    # get p val
    p_val = np.min([np.sum(asymm_null >= observed) / n_perms,
                    np.sum(asymm_null <= observed) / n_perms])

    return asymm_null, observed, p_val

def helper_null_hyperplane(e, e_null, indices, param='r2'):
    n_perms = e_null.shape[2]
    # compute energy asymmetry
    ed = e.transpose() - e

    # containers
    ed_null = np.zeros(e_null.shape)
    asymm_null = np.zeros(n_perms)

    for i in np.arange(e_null.shape[2]):
        # compute null asymmetry matrix
        ed_null[:, :, i] = e_null[:, :, i].transpose() - e_null[:, :, i]

        data = np.concatenate((indices[0].reshape(-1, 1),
                               indices[1].reshape(-1, 1),
                               ed_null[:, :, i][indices].reshape(-1, 1)), axis=1)
        data = (data - data.mean(axis=0)) / data.std(axis=0)
        _, _, _, c, r2 = fit_hyperplane(data)
        if param == 'r2':
            asymm_null[i] = r2
        elif param == 'cx':
            asymm_null[i] = c[0]
        elif param == 'cy':
            asymm_null[i] = c[1]

    # get observed
    data = np.concatenate((indices[0].reshape(-1, 1), indices[1].reshape(-1, 1), ed[indices].reshape(-1, 1)), axis=1)
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    _, _, _, c, r2 = fit_hyperplane(data)

    # get p val
    if param == 'r2':
        observed = r2
        p_val = np.min([np.sum(asymm_null >= observed) / n_perms,
                        np.sum(asymm_null <= observed) / n_perms])
    elif param == 'cx':
        observed = c[0]
        p_val = np.min([np.sum(np.abs(asymm_null) >= np.abs(observed)) / n_perms,
                        np.sum(np.abs(asymm_null) <= np.abs(observed)) / n_perms])
    elif param == 'cy':
        observed = c[1]
        p_val = np.min([np.sum(np.abs(asymm_null) >= np.abs(observed)) / n_perms,
                        np.sum(np.abs(asymm_null) <= np.abs(observed)) / n_perms])

    return asymm_null, observed, p_val

# %% plots

# data for plotting
B = 'wb'
e = E[B] # energy matrix
ed = e.transpose() - e # energy asymmetry matrix

network_null = 'wwp'
file = 'average_adj_n-{0}_s-{1}_null-mni-{2}_ns-{3}-0_c-minimum_fast_T-1_B-{4}_E.npy'.format(n_subs, spars_thresh,
                                                                                             network_null, n_states, B)
e_null = np.load(os.path.join(environment.pipelinedir, 'minimum_control_energy', file))

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
textstr = 't = {:.2f}; p = {:.2f}'.format(t, p_val)
ax.text(0.025, 0.95, textstr, transform=ax.transAxes, style='italic',
        verticalalignment='top', rotation='horizontal')
f.savefig(os.path.join(environment.figdir, 'energy_{0}.png'.format(B)), dpi=300, bbox_inches='tight',
          pad_inches=0.1)
plt.close()
