import os
import numpy as np
import scipy as sp
import pandas as pd
from pfactor_gradients.pnc import Environment, Subject
from pfactor_gradients.routines import LoadSC, LoadCT, LoadRLFP, LoadCBF, LoadREHO, LoadALFF, LoadAverageBrainMaps
from pfactor_gradients.pipelines import ComputeGradients
from pfactor_gradients.utils import get_null_p, get_fdr_p
from tqdm import tqdm

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
compute_gradients = ComputeGradients(environment=environment, Subject=Subject)
compute_gradients.run()

# %% Load sc data
load_sc = LoadSC(environment=environment, Subject=Subject)
load_sc.run()
# refilter environment due to LoadSC excluding on disconnected nodes
environment.df = load_sc.df.copy()

# %% load energy and null
n_subs = environment.df.shape[0]
n_states = len(np.unique(compute_gradients.grad_bins))
mask = ~np.eye(n_states, dtype=bool)
indices = np.where(mask)

if parc == 'schaefer' and n_parcels == 400:
    sparse_thresh = 0.06
elif parc == 'schaefer' and n_parcels == 200:
    sparse_thresh = 0.12

n_perms = 1000
my_list = ['ct', 'cbf', 'reho', 'alff', 'wb']

for B in my_list:
    file = 'average_adj_n-{0}_s-{1}_ns-{2}-0_c-minimum_fast_T-1_B-{3}_E.npy'.format(n_subs, sparse_thresh, n_states, B)
    E = np.load(os.path.join(environment.pipelinedir, 'minimum_control_energy', file))

    E_null = np.zeros((n_states, n_states, n_perms))
    for i in tqdm(np.arange(n_perms)):
        file = 'average_adj_n-{0}_s-{1}_null-mni-{2}-{3}_ns-{4}-0_c-minimum_fast_T-1_B-{5}_E.npy'.format(n_subs,
                                                                                                         sparse_thresh,
                                                                                                         'wwp', i,
                                                                                                         n_states, B)
        E_null[:, :, i] = np.load(os.path.join(environment.pipelinedir, 'minimum_control_energy', file))

    energy_delta = np.zeros((n_states, n_states))
    p_vals = np.zeros((n_states, n_states))
    for i in np.arange(n_states):
        for j in np.arange(n_states):
            energy_delta[i, j] = E[i, j] - np.mean(E_null[i, j, :])
            # p_vals[i, j] = np.sum(E[i, j] >= E_null[i, j, :]) / n_perms)
            # p_vals[i, j] = np.sum(E[i, j] < E_null[i, j, :]) / n_perms)
            p_vals[i, j] = np.min([np.sum(E[i, j] >= E_null[i, j, :]) / n_perms,
                                   np.sum(E[i, j] < E_null[i, j, :]) / n_perms])

    # p_vals = get_fdr_p(p_vals)
    sig_mask = p_vals > 0.05
    print(np.sum(sig_mask==False))

    f, ax = plt.subplots(1, 1, figsize=(2.5, 2.5))
    # sns.heatmap(p_vals, mask=sig_mask, square=True, ax=ax)
    sns.heatmap(energy_delta, mask=sig_mask, square=True, ax=ax, center=0, vmin=-1, vmax=1, cmap='coolwarm')
    ax.set_title(B)
    f.savefig(os.path.join(environment.figdir, 'fig-4a_nulls_{0}.png'.format(B)), dpi=150, bbox_inches='tight',
              pad_inches=0.1)
    plt.close()
