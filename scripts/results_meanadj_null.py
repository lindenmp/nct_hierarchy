import os
import numpy as np
import scipy as sp
import pandas as pd
from pfactor_gradients.pnc import Environment, Subject
from pfactor_gradients.imaging_derivs import DataMatrix, DataVector
from pfactor_gradients.pipelines import ComputeGradients
from pfactor_gradients.plotting import my_regplot
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

# %% load energy and null
def get_snull_file(null_model='wwp', sge_task_id=0, B='wb'):
    return 'average_adj_n-775_s-0.06_null-mni-{0}-{1}_ns-20-20_c-minimum_fast_T-1_B-{2}_E.npy'.format(null_model,
                                                                                                      sge_task_id, B)
def get_bnull_file(sge_task_id=0, B='wb'):
    return 'average_adj_n-775_s-0.06_ns-20-20_c-minimum_fast_T-1_B-{0}-null-{1}_E.npy'.format(B, sge_task_id)

B = 'ct'
file_name = 'average_adj_n-775_s-0.06_ns-20-20_c-minimum_fast_T-1_B-{0}_E.npy'.format(B)
E = np.load(os.path.join(environment.pipelinedir, 'minimum_control_energy', file_name))

n_perms = 5000
n_states = E.shape[0]

E_net_null = np.zeros((n_states, n_states, n_perms))
E_brain_null = np.zeros((n_states, n_states, n_perms))

for i in tqdm(np.arange(n_perms)):
    file_name = get_snull_file(null_model='wwp', sge_task_id=i, B=B)
    E_net_null[:, :, i] = np.load(os.path.join(environment.pipelinedir, 'minimum_control_energy', file_name))

    file_name = get_bnull_file(B=B, sge_task_id=i)
    E_brain_null[:, :, i] = np.load(os.path.join(environment.pipelinedir, 'minimum_control_energy', file_name))

# %%
p_vals_net = np.zeros((n_states, n_states))
p_vals_brain = np.zeros((n_states, n_states))

for i in np.arange(n_states):
    for j in np.arange(n_states):
        p_vals_net[i, j] = get_null_p(E[i, j], E_net_null[i, j, :])
        p_vals_brain[i, j] = get_null_p(E[i, j], E_brain_null[i, j, :])

p_vals_net = get_fdr_p(p_vals_net)
sig_mask_net = p_vals_net > 0.05
print(np.sum(sig_mask_net==False))

p_vals_brain = get_fdr_p(p_vals_brain)
sig_mask_brain = p_vals_brain > 0.05
print(np.sum(sig_mask_brain==False))

f, ax = plt.subplots(1, 2, figsize=(10, 5))
sns.heatmap(p_vals_net, mask=sig_mask_net, square=True, ax=ax[0])
ax[0].set_title('Network model null')
sns.heatmap(p_vals_brain, mask=sig_mask_brain, square=True, ax=ax[1])
ax[1].set_title('Brain map null')
# sns.heatmap(p_vals, square=True, ax=ax)
f.savefig(os.path.join(environment.figdir, 'nulls_{0}.png'.format(B)), dpi=150, bbox_inches='tight',
          pad_inches=0.1)
plt.close()
