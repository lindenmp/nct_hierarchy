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
    load_average_bms.brain_maps[key].mean_between_states(compute_gradients.grad_bins)

n_clusters = len(np.unique(compute_gradients.grad_bins))
mask = ~np.eye(n_clusters, dtype=bool)
indices = np.where(mask)

# %% load energy and null
def get_null_file(null_model='wwp', sge_task_id=0, B='wb'):
    return 'average_adj_n-775_s-0.06_null-mni-{0}-{1}_ns-40-0_c-minimum_fast_T-1_B-{2}_E.npy'.format(null_model,
                                                                                                      sge_task_id, B)
def get_spin_file(sge_task_id=0, B='wb'):
    return 'average_adj_n-775_s-0.06_ns-40-0_c-minimum_fast_T-1_B-{0}-spin-{1}_E.npy'.format(B, sge_task_id)

def get_rand_file(sge_task_id=0, B='wb'):
    return 'average_adj_n-775_s-0.06_ns-40-0_c-minimum_fast_T-1_B-{0}-rand-{1}_E.npy'.format(B, sge_task_id)

# %% load energy and null
for key in load_average_bms.brain_maps:
    file_name = 'average_adj_n-775_s-0.06_ns-40-0_c-minimum_fast_T-1_B-{0}_E.npy'.format(key)
    E = np.load(os.path.join(environment.pipelinedir, 'minimum_control_energy', file_name))

    n_perms = 1000
    n_states = E.shape[0]

    E_null = np.zeros((n_states, n_states, n_perms))
    E_spin_null = np.zeros((n_states, n_states, n_perms))
    E_rand_null = np.zeros((n_states, n_states, n_perms))

    for i in tqdm(np.arange(n_perms)):
        file_name = get_null_file(null_model='wwp', sge_task_id=i, B=key)
        E_null[:, :, i] = np.load(os.path.join(environment.pipelinedir, 'minimum_control_energy', file_name))

        file_name = get_spin_file(B=key, sge_task_id=i)
        E_spin_null[:, :, i] = np.load(os.path.join(environment.pipelinedir, 'minimum_control_energy', file_name))

        file_name = get_rand_file(B=key, sge_task_id=i)
        E_rand_null[:, :, i] = np.load(os.path.join(environment.pipelinedir, 'minimum_control_energy', file_name))

    p_vals_net = np.zeros((n_states, n_states))
    p_vals_brain = np.zeros((n_states, n_states))
    energy_delta = np.zeros((n_states, n_states))

    for i in np.arange(n_states):
        for j in np.arange(n_states):
            p_vals_net[i, j] = get_null_p(E[i, j], E_null[i, j, :])
            p_vals_brain[i, j] = get_null_p(E[i, j], E_spin_null[i, j, :])
            # p_vals_brain[i, j] = get_null_p(E[i, j], E_rand_null[i, j, :])
            energy_delta[i, j] = E[i, j] - np.mean(E_spin_null[i, j, :])

    # p_vals_net = get_fdr_p(p_vals_net)
    sig_mask_net = p_vals_net > 0.05
    print(np.sum(sig_mask_net==False))

    # p_vals_brain = get_fdr_p(p_vals_brain)
    sig_mask_brain = p_vals_brain > 0.05
    print(np.sum(sig_mask_brain==False))

    f, ax = plt.subplots(1, 2, figsize=(10, 5))
    sns.heatmap(p_vals_net, mask=sig_mask_net, square=True, ax=ax[0])
    ax[0].set_title('Network model null')
    # sns.heatmap(p_vals_brain, mask=sig_mask_brain, square=True, ax=ax[1])
    sns.heatmap(energy_delta, square=True, ax=ax[1])
    ax[1].set_title('Brain map null')
    # sns.heatmap(p_vals, square=True, ax=ax)
    f.savefig(os.path.join(environment.figdir, 'nulls_{0}.png'.format(key)), dpi=150, bbox_inches='tight',
              pad_inches=0.1)
    plt.close()

# i = 2
# j = 10
# f, ax = plt.subplots(1, 1, figsize=(5, 5))
# sns.histplot(x=E_null[i, j, :], ax=ax)
# ax.axvline(E[i, j], color='r')
# textstr = 'p unc. = {:.2f}'.format(p_vals[i, j])
# ax.text(0.01, 0.975, textstr, transform=ax.transAxes,
#         verticalalignment='top', rotation='horizontal', c='r')
#
# f.savefig(os.path.join(environment.figdir, 'test.png'), dpi=150, bbox_inches='tight',
#           pad_inches=0.1)
# plt.close()
