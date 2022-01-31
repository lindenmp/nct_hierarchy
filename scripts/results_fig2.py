# %% import
import sys, os, platform
from pfactor_gradients.imaging_derivs import DataMatrix
from pfactor_gradients.pipelines import ComputeMinimumControlEnergy
from pfactor_gradients.plotting import my_reg_plot, my_distpair_plot, my_null_plot
from pfactor_gradients.utils import rank_int, get_null_p

import scipy as sp
from tqdm import tqdm

# %% import workspace
os.environ["MY_PYTHON_WORKSPACE"] = 'ave_adj'
os.environ["WHICH_BRAIN_MAP"] = 'hist-g2'
# os.environ["WHICH_BRAIN_MAP"] = 'func-g1'
from setup_workspace import *

# %% plotting
import seaborn as sns
import matplotlib.pyplot as plt
from pfactor_gradients.plotting import set_plotting_params
set_plotting_params(format='svg')
figsize = 1.5

# %% get control energy
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
    n_perms = 5000
    # network_null = 'mni-wwp'
    network_null = 'mni-wsp'
    e_network_null = np.zeros((n_states, n_states, n_perms))

    for i in tqdm(np.arange(n_perms)):
        file = 'average_adj_n-{0}_cthr-{1}_smap-{2}_null-{3}-{4}_ns-{5}_ctrl-minimum_fast_c-{6}_T-{7}_B-{8}_E.npy' \
            .format(n_subs,
                    consist_thresh,
                    which_brain_map,
                    network_null,
                    i, n_states, c, T, B)
        e_network_null[:, :, i] = np.load(os.path.join(environment.pipelinedir, 'minimum_control_energy', file))

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

# plot distance asymm
f, ax = plt.subplots(1, 1, figsize=(figsize, figsize))
my_reg_plot(states_distance[indices_upper], ed[indices_upper],
                        'hierarchy distance', 'energy asymmetry', ax, annotate='spearman')
f.savefig(os.path.join(environment.figdir, 'corr(distance,e_asym_{0})'.format(B)), dpi=600, bbox_inches='tight',
          pad_inches=0.01)
plt.close()

# plot null
try:
    r_null = np.zeros(n_perms)

    for i in np.arange(n_perms):
        ed_null = e_network_null[:, :, i] - e_network_null[:, :, i].transpose()
        r_null[i] = sp.stats.spearmanr(states_distance[indices_upper], ed_null[indices_upper])[0]

    # plot distance asymm null
    observed = sp.stats.spearmanr(states_distance[indices_upper], ed[indices_upper])[0]
    p_val = get_null_p(observed, r_null, abs=True)
    f, ax = plt.subplots(1, 1, figsize=(figsize, figsize))
    my_null_plot(observed=observed, null=r_null, p_val=p_val, xlabel='hierarchy distance \ncorrelation (Rho)', ax=ax)
    f.savefig(os.path.join(environment.figdir, 'corr(distance,e_asym_{0})_null_{1}'.format(B, network_null)), dpi=600,
              bbox_inches='tight', pad_inches=0.01)
    plt.close()
except NameError:
    print('Requisite variables not found...')
