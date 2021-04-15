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

sns.set(style='white', context='talk', font_scale=1)
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
load_average_bms.run()

for key in load_average_bms.brain_maps:
    load_average_bms.brain_maps[key].mean_between_clusters(compute_gradients.grad_bins)

n_states = len(np.unique(compute_gradients.grad_bins))
mask = ~np.eye(n_states, dtype=bool)
indices = np.where(mask)

# %% load energy and null
n_perms = 1000
my_list = ['wb',] + list(load_average_bms.brain_maps.keys())

for B in my_list:
    file_name = 'average_adj_n-775_s-0.06_ns-40-0_c-minimum_fast_T-1_B-{0}_E.npy'.format(B)
    E = np.load(os.path.join(environment.pipelinedir, 'minimum_control_energy', file_name))

    E_null = np.zeros((n_states, n_states, n_perms))
    for i in tqdm(np.arange(n_perms)):
        file_name = 'average_adj_n-775_s-0.06_null-mni-{0}-{1}_ns-40-0_c-minimum_fast_T-1_B-{2}_E.npy'\
            .format('wwp', i, B)
        E_null[:, :, i] = np.load(os.path.join(environment.pipelinedir, 'minimum_control_energy', file_name))

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

    f, ax = plt.subplots(1, 1, figsize=(5, 5))
    # sns.heatmap(p_vals, mask=sig_mask, square=True, ax=ax)
    sns.heatmap(energy_delta, mask=sig_mask, square=True, ax=ax)
    ax.set_title(B)
    f.savefig(os.path.join(environment.figdir, 'nulls_{0}.png'.format(B)), dpi=150, bbox_inches='tight',
              pad_inches=0.1)
    plt.close()
