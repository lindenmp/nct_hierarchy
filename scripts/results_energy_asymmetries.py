import sys, os, platform
from pfactor_gradients.pnc import Environment, Subject
from pfactor_gradients.routines import LoadSC, LoadCT, LoadRLFP, LoadCBF, LoadREHO, LoadALFF,\
    LoadAverageSC, LoadAverageBrainMaps
from pfactor_gradients.pipelines import ComputeGradients, ComputeMinimumControlEnergy
from pfactor_gradients.utils import rank_int, get_fdr_p
from pfactor_gradients.plotting import my_regplot
import numpy as np
import pandas as pd
import scipy as sp

# %% Plotting
import matplotlib.pyplot as plt
import seaborn as sns

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
E = dict()

nct_pipeline = ComputeMinimumControlEnergy(environment=environment, A=load_average_sc.A,
                                           states=compute_gradients.grad_bins, n_subsamples=n_subsamples,
                                           control='minimum_fast', T=1, B='wb', file_prefix=file_prefix,
                                           force_rerun=False, save_outputs=True, verbose=True)
nct_pipeline.run()
E['wb'] = nct_pipeline.E

for i, key in enumerate(load_average_bms.brain_maps):
    nct_pipeline = ComputeMinimumControlEnergy(environment=environment, A=A,
                                               states=compute_gradients.grad_bins, n_subsamples=n_subsamples,
                                               control='minimum_fast', T=1, B=load_average_bms.brain_maps[key], file_prefix=file_prefix,
                                               force_rerun=False, save_outputs=True, verbose=True)
    nct_pipeline.run()

    E[key] = nct_pipeline.E

# %% plots

# %% energy asymmetry
B = 'wb'
e = E[B]
print(np.var(e[np.eye(n_states).astype(bool)]))
# e = np.log(e)
# e = np.sqrt(e)
# e = rank_int(e)

# e_self1 = np.tile(e[np.eye(n_states).astype(bool)], reps=(n_states, 1))
# e_self1[np.tril_indices(n_states)] = 0
# e_self2 = np.tile(e[np.eye(n_states).astype(bool)], reps=(n_states, 1)).transpose()
# e_self2[np.triu_indices(n_states)] = 0
# e_self = e_self1 + e_self2 + np.eye(n_states)

e_self = np.tile(e[np.eye(n_states).astype(bool)], reps=(n_states, 1))
# e_self = np.tile(e[np.eye(n_states).astype(bool)], reps=(n_states, 1)).transpose()
# e = np.divide(e, e_self)

# e = np.log(e)
# e = np.sqrt(e)
e = rank_int(e)

eu = e[indices_upper]
el = e[indices_lower]
print(np.nanmedian(eu), np.nanmedian(el), np.nanmedian(eu) - np.nanmedian(el))
print(sp.stats.wilcoxon(x=eu, y=el, alternative='less'))
print(np.nanmean(eu), np.nanmean(el), np.nanmean(eu) - np.nanmean(el))
print(sp.stats.ttest_rel(a=eu, b=el))

df_plot = pd.DataFrame(data=np.vstack((eu, el)).transpose(), columns=['bottom-up', 'top-down'])
ed = e - e.transpose()
plot_mask = np.zeros((n_states, n_states))
plot_mask[indices_lower] = 1
plot_mask = plot_mask.astype(bool)

f, ax = plt.subplots(1, 3, figsize=(9, 3))
sns.violinplot(data=df_plot, ax=ax[0])
sns.heatmap(e, center=0, square=True, cmap='coolwarm', ax=ax[1])
sns.heatmap(ed, mask=plot_mask, center=0, square=True, cmap='coolwarm', ax=ax[2])
# sns.heatmap(ed, mask=plot_mask, center=0, vmin=-0.4, vmax=0.4, square=True, cmap='coolwarm', ax=ax[1])
plt.subplots_adjust(wspace=.25)
f.savefig(os.path.join(environment.figdir, 'energy_asymmetry_{0}.png'.format(B)), dpi=150, bbox_inches='tight',
          pad_inches=0.1)
plt.close()

# %% energy asymmetry vs effective connectivity asymmetry

# load dcm outputs
file = 'dcm_ns-{0}_A.mat'.format(n_states)
dcm = sp.io.loadmat(os.path.join(environment.pipelinedir, 'spdcm', file))
# ec = dcm['Ep']['A'][0][0]
ec = dcm['A']
# ecd = ec - ec.transpose()
ecd = np.abs(ec) - np.abs(ec).transpose()

f, ax = plt.subplots(1, 2, figsize=(6, 3))
sns.heatmap(ec, center=0, vmin=-1, vmax=1, square=True, ax=ax[0])
ax[0].tick_params(pad=-2.5)
sns.heatmap(ecd, center=0, vmin=-1, vmax=1, square=True, ax=ax[1])
ax[1].tick_params(pad=-2.5)
plt.subplots_adjust(wspace=.25)
f.savefig(os.path.join(environment.figdir, 'ec.png'), dpi=150, bbox_inches='tight',
          pad_inches=0.1)
plt.close()

f, ax = plt.subplots(1, 1, figsize=(3, 3))
my_regplot(x=ed[indices_upper], y=ecd[indices_upper], xlabel='energy (delta)', ylabel='effective connectivity (delta)', ax=ax)
plt.subplots_adjust(wspace=.25)
f.savefig(os.path.join(environment.figdir, 'ed_ecd_{0}.png'.format(B)), dpi=150, bbox_inches='tight',
          pad_inches=0.1)
plt.close()

# %% energy null network model
def null_helper(e, e_null):
    n_states = e.shape[0]
    n_perms = e_null.shape[2]

    e_null_delta = np.zeros((n_states, n_states))
    p_vals = np.zeros((n_states, n_states))

    for i in np.arange(n_states):
        for j in np.arange(n_states):
            e_null_delta[i, j] = e[i, j] - np.mean(e_null[i, j, :])
            p_vals[i, j] = np.min([np.sum(e[i, j] >= e_null[i, j, :]) / n_perms,
                                   np.sum(e[i, j] < e_null[i, j, :]) / n_perms])

    return e_null_delta, p_vals

B_list = ['wb',] + list(load_average_bms.brain_maps.keys())
null_list = ['wwp', 'wsp', 'wssp']

for B in B_list:
    for network_null in null_list:
        e = E[B]

        # load null energy
        file = 'average_adj_n-{0}_s-{1}_null-mni-{2}_ns-{3}-0_c-minimum_fast_T-1_B-{4}_E.npy'.format(n_subs, spars_thresh,
                                                                                                     network_null, n_states, B)
        e_null = np.load(os.path.join(environment.pipelinedir, 'minimum_control_energy', file))
        e_null_delta, p_vals = null_helper(e, e_null)

        p_vals = get_fdr_p(p_vals)
        sig_mask = p_vals > 0.05
        print(np.sum(sig_mask==False))

        f, ax = plt.subplots(1, 1, figsize=(3, 3))
        sns.heatmap(e_null_delta, mask=sig_mask, square=True, ax=ax, center=0, cmap='coolwarm')
        ax.set_title(B)
        ax.tick_params(pad=-2.5)
        f.savefig(os.path.join(environment.figdir, 'e_network_null_{0}_{1}.png'.format(B, network_null)), dpi=150,
                  bbox_inches='tight', pad_inches=0.1)
        plt.close()
