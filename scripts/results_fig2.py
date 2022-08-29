# %% import
import sys, os, platform
from src.imaging_derivs import DataMatrix
from src.pipelines import ComputeMinimumControlEnergy
from src.plotting import my_reg_plot, my_distpair_plot, my_null_plot, my_bsci_plot
from src.utils import rank_int, get_null_p, mean_over_states

import pandas as pd
import scipy as sp
from tqdm import tqdm

# %% import workspace
os.environ["MY_PYTHON_WORKSPACE"] = 'ave_adj'
os.environ["WHICH_BRAIN_MAP"] = 'hist-g2'
# os.environ["WHICH_BRAIN_MAP"] = 'micro-g1'
# os.environ["WHICH_BRAIN_MAP"] = 'func-g1'
# os.environ["WHICH_BRAIN_MAP"] = 'myelin'
os.environ["INTRAHEMI"] = "False"
from setup_workspace import *

# %% plotting
import seaborn as sns
import matplotlib.pyplot as plt
from src.plotting import set_plotting_params
set_plotting_params(format='svg')
figsize = 1.5

# %%
if intrahemi == False:
    dv = DataVector(data=state_brain_map, name=which_brain_map)
    dv.rankdata()
    dv.brain_surface_plot(environment)

# %% plot mean adjacency matrix
f, ax = plt.subplots(1, 1, figsize=(figsize*1.2, figsize*1.2))
sns.heatmap(A, square=True, ax=ax, cbar_kws={"shrink": 0.80})
ax.tick_params(pad=-2.5)
f.savefig(os.path.join(environment.figdir, 'A.png'), dpi=600, bbox_inches='tight', pad_inches=0.01)
plt.close()

# %% get control energy
if intrahemi == True:
    file_prefix = 'average_adj_n-{0}_intrahemi_cthr-{1}_smap-{2}_'.format(load_average_sc.load_sc.df.shape[0],
                                                                    consist_thresh, which_brain_map)
    n_parcels = int(n_parcels / 2)
else:
    file_prefix = 'average_adj_n-{0}_cthr-{1}_smap-{2}_'.format(load_average_sc.load_sc.df.shape[0],
                                                                consist_thresh, which_brain_map)

B_dict = dict()
B = DataMatrix(data=np.eye(n_parcels), name='identity')
B_dict[B.name] = B

c = 1
T = 1
E = dict.fromkeys(B_dict)

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
B = 'identity'
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

# load permuted data
try:
    network_null = 'mni-wwp'
    # network_null = 'mni-wsp'
    file = 'average_adj_n-{0}_cthr-{1}_smap-{2}_null-{3}-E.npy'.format(load_average_sc.load_sc.df.shape[0],
                                                                   consist_thresh, which_brain_map,
                                                                   network_null)
    e_network_null = np.load(os.path.join(environment.pipelinedir, 'minimum_control_energy', file))

    # normalize
    n_perms = e_network_null.shape[2]
    if norm_energy:
        for i in tqdm(np.arange(n_perms)):
            e_network_null[:, :, i] = rank_int(e_network_null[:, :, i])

except FileNotFoundError:
    print('Requisite files not found...')

# load bootstrapped data
try:
    file = 'average_adj_n-{0}_cthr-{1}_smap-{2}_bootstrapped_E'.format(load_average_sc.load_sc.df.shape[0],
                                                                       consist_thresh, which_brain_map)

    e_bs = np.load(os.path.join(environment.pipelinedir, 'minimum_control_energy', file + '.npy'))

    # normalize
    n_samples = e_bs.shape[2]
    if norm_energy:
        for i in tqdm(np.arange(n_samples)):
            e_bs[:, :, i] = rank_int(e_bs[:, :, i])

except FileNotFoundError:
    print('Requisite files not found...')

# %% energy matrix
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
f.savefig(os.path.join(environment.figdir, 'e'), dpi=600, bbox_inches='tight', pad_inches=0.01)
plt.close()

# %% Panel A: energy asymmetry

# top-down vs bottom-up
f, ax = plt.subplots(1, 1, figsize=(figsize, figsize))
df_plot = pd.DataFrame(data=np.vstack((e[indices_upper], e.transpose()[indices_upper])).transpose(),
                       columns=['bottom-up', 'top-down'])
my_distpair_plot(df=df_plot, ylabel='energy (z-score)', ax=ax, test_stat='mean_diff')
f.savefig(os.path.join(environment.figdir, 'e_asym_dists'), dpi=600, bbox_inches='tight',
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
f.savefig(os.path.join(environment.figdir, 'e_asym_matrix'), dpi=600, bbox_inches='tight', pad_inches=0.01)
plt.close()

# plot bootstrap
try:
    ed_mean_bs = np.zeros(n_samples)

    for i in np.arange(n_samples):
        ed_strap = e_bs[:, :, i] - e_bs[:, :, i].transpose()  # energy asymmetry matrix
        ed_mean_bs[i] = np.mean(ed_strap[indices_upper])

    f, ax = plt.subplots(1, 1, figsize=(figsize, figsize))
    my_bsci_plot(dist=ed_mean_bs, observed=np.mean(ed[indices_upper]), xlabel='mean asymmetry\n(bootstrap)', ax=ax)
    f.savefig(os.path.join(environment.figdir, 'e_asym_bootstrap'), dpi=600,
              bbox_inches='tight', pad_inches=0.01)
    plt.close()
except NameError:
    print('Requisite variables not found...')

# plot null
try:
    t_null = np.zeros(n_perms)

    for i in np.arange(n_perms):
        ed_null = e_network_null[:, :, i] - e_network_null[:, :, i].transpose()
        t_null[i] = np.mean(ed_null[indices_upper])

    # plot distance asymm null
    observed = np.mean(ed[indices_upper])
    p_val = get_null_p(observed, t_null, version='standard', abs=True)
    f, ax = plt.subplots(1, 1, figsize=(figsize, figsize))
    my_null_plot(observed=observed, null=t_null, p_val=p_val, xlabel='mean asymmetry', ax=ax)
    f.savefig(os.path.join(environment.figdir, 'e_asym_null'), dpi=600,
              bbox_inches='tight', pad_inches=0.01)
    plt.close()
except NameError:
    print('Requisite variables not found...')

# %% Panel B: energy asymmetry distance corr

# get hierarchy distance between states
states_distance = mean_over_states(state_brain_map, states)
states_distance = np.abs(states_distance)

# plot distance asymm
f, ax = plt.subplots(1, 1, figsize=(figsize, figsize))
my_reg_plot(states_distance[indices_upper], ed[indices_upper],
                        'hierarchy distance', 'energy asymmetry', ax, annotate='spearman')
f.savefig(os.path.join(environment.figdir, 'corr(distance,e_asym)'), dpi=600, bbox_inches='tight',
          pad_inches=0.01)
plt.close()

# plot bootstrap
try:
    r_bs = np.zeros(n_samples)

    for i in np.arange(n_samples):
        ed_strap = e_bs[:, :, i] - e_bs[:, :, i].transpose()  # energy asymmetry matrix
        r_bs[i] = sp.stats.spearmanr(states_distance[indices_upper], ed_strap[indices_upper])[0]

    observed = sp.stats.spearmanr(states_distance[indices_upper], ed[indices_upper])[0]
    f, ax = plt.subplots(1, 1, figsize=(figsize, figsize))
    my_bsci_plot(dist=r_bs, observed=observed, xlabel='hierarchy distance correlation\n(bootstrap)', ax=ax)
    f.savefig(os.path.join(environment.figdir, 'corr(distance,e_asym)_bootstrap'), dpi=600,
              bbox_inches='tight', pad_inches=0.01)
    plt.close()
except NameError:
    print('Requisite variables not found...')

# plot null
try:
    r_null = np.zeros(n_perms)

    for i in np.arange(n_perms):
        ed_null = e_network_null[:, :, i] - e_network_null[:, :, i].transpose()
        r_null[i] = sp.stats.spearmanr(states_distance[indices_upper], ed_null[indices_upper])[0]

    # plot distance asymm null
    observed = sp.stats.spearmanr(states_distance[indices_upper], ed[indices_upper])[0]
    p_val = get_null_p(observed, r_null, version='standard', abs=True)
    f, ax = plt.subplots(1, 1, figsize=(figsize, figsize))
    my_null_plot(observed=observed, null=r_null, p_val=p_val, xlabel='hierarchy distance \ncorrelation (Rho)', ax=ax)
    f.savefig(os.path.join(environment.figdir, 'corr(distance,e_asym)_null'), dpi=600,
              bbox_inches='tight', pad_inches=0.01)
    plt.close()
except NameError:
    print('Requisite variables not found...')
