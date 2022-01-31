# %% import
import sys, os, platform
from pfactor_gradients.imaging_derivs import DataMatrix
from pfactor_gradients.pipelines import ComputeMinimumControlEnergy
from pfactor_gradients.plotting import my_reg_plot, my_distpair_plot, my_null_plot
from pfactor_gradients.energy import expand_states
from pfactor_gradients.utils import rank_int, get_null_p, get_bootstrap_indices, mean_confidence_interval

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
# B = 'identity'
B = 'E_opt'
print(B)
e = E[B].copy()

if norm_energy:
    e = rank_int(e) # normalized energy matrix

ed = e - e.transpose() # energy asymmetry matrix
print(np.all(np.round(np.abs(ed.flatten()), 4) == np.round(np.abs(ed.transpose().flatten()), 4)))

# save out mean ed for use in other scripts
np.save(os.path.join(environment.pipelinedir, 'e_{0}_{1}.npy'.format(which_brain_map, B)), e)
np.save(os.path.join(environment.pipelinedir, 'ed_{0}_{1}.npy'.format(which_brain_map, B)), ed)

# %% Panel B: energy asymmetry
# top-down vs bottom-up
f, ax = plt.subplots(1, 1, figsize=(figsize, figsize))
df_plot = pd.DataFrame(data=np.vstack((e[indices_upper], e[indices_lower])).transpose(),
                       columns=['bottom-up', 'top-down'])
my_distpair_plot(df=df_plot, ylabel='energy (z-score)', ax=ax)
f.savefig(os.path.join(environment.figdir, 'e_asym_{0}_dists'.format(B)), dpi=600, bbox_inches='tight',
          pad_inches=0.01)
plt.close()

# %% Panel C: bootstrap
n_samples = 1000
bootstrap_indices = get_bootstrap_indices(d_size=n_subs, n_samples=n_samples)

ed_bs = np.zeros(n_samples)
ed_opt_bs = np.zeros(n_samples)

# loop start
for i in tqdm(np.arange(n_samples)):
    file_prefix = 'average_adj_n-{0}_cthr-{1}_smap-{2}_strap-{3}_'.format(load_average_sc.load_sc.df.shape[0],
                                                                consist_thresh, which_brain_map, i)

    load_sc_strap = LoadSC(environment=environment, Subject=Subject)
    load_sc_strap.df = load_sc.df.iloc[bootstrap_indices[i, :], :]
    load_sc_strap.A = load_sc.A[:, :, bootstrap_indices[i, :]]

    load_average_sc_strap = LoadAverageSC(load_sc=load_sc_strap, consist_thresh=consist_thresh, verbose=False)
    load_average_sc_strap.run()

    # get bootstrapped energy
    nct_pipeline = ComputeMinimumControlEnergy(environment=environment, A=load_average_sc_strap.A, states=states, B=B_dict['identity'],
                                               control='minimum_fast', c=c, T=T,
                                               file_prefix=file_prefix,
                                               force_rerun=False, save_outputs=True, verbose=False)
    n = 2
    ds = 0.1
    nct_pipeline.run_with_optimized_b(n=n, ds=ds)

    e_strap = nct_pipeline.E_opt[:, 0].reshape(n_states, n_states)
    e_strap = rank_int(e_strap)  # normalized energy matrix
    ed_strap = e_strap.transpose() - e_strap  # energy asymmetry matrix
    ed_strap = np.mean(ed_strap[indices_upper])
    ed_bs[i] = ed_strap

    e_opt_strap = nct_pipeline.E_opt[:, 1].reshape(n_states, n_states)
    e_opt_strap = rank_int(e_opt_strap)  # normalized energy matrix
    ed_opt_strap = e_opt_strap.transpose() - e_opt_strap  # energy asymmetry matrix
    ed_opt_strap = np.mean(ed_opt_strap[indices_upper])
    ed_opt_bs[i] = ed_opt_strap

# plot
f, ax = plt.subplots(1, 1, figsize=(figsize*0.75, figsize))
df_plot = pd.DataFrame(data=np.vstack((ed_bs, ed_opt_bs)).transpose(),
                       columns=['identity', 'optimized'])
my_distpair_plot(df=df_plot, ylabel='energy delta', ax=ax, test_stat=None, split=True)
f.savefig(os.path.join(environment.figdir, 'e_asym_dists_bootstrap'), dpi=600, bbox_inches='tight',
          pad_inches=0.01)
plt.close()

print(np.round(mean_confidence_interval(data=ed_bs), 4))
print(np.round(mean_confidence_interval(data=ed_opt_bs), 4))

# %% Panel D: null for brain map spatial corrs
x0_mat, xf_mat = expand_states(states)
n_transitions = x0_mat.shape[1]

n_perms = 5000
# network_null = 'mni-wwp'
network_null = 'mni-wsp'

try:
    r_null = np.load(os.path.join(environment.pipelinedir, 'optimized_weights_r_null_{0}_{1}.npy' \
                                  .format(which_brain_map, network_null)))
    p_vals = np.load(os.path.join(environment.pipelinedir, 'optimized_weights_p_vals_{0}_{1}.npy' \
                                  .format(which_brain_map, network_null)))
    observed = np.load(os.path.join(environment.pipelinedir, 'optimized_weights_observed_{0}_{1}.npy' \
                                  .format(which_brain_map, network_null)))
except:
    B_opt_network_null = np.zeros((n_parcels, n_transitions, n_perms))

    for i in tqdm(np.arange(n_perms)):
        file = 'average_adj_n-{0}_cthr-{1}_smap-{2}_null-{3}-{4}_ns-{5}_ctrl-minimum_fast_c-{6}_T-{7}_B-optimized-n-2-ds-0.1_weights.npy' \
            .format(n_subs,
                    consist_thresh,
                    which_brain_map,
                    network_null,
                    i, n_states, c, T)
        B_opt_network_null[:, :, i] = np.load(os.path.join(environment.pipelinedir, 'minimum_control_energy', file))[:, :, 1]

    observed = np.zeros(n_transitions)
    r_null = np.zeros((n_transitions, n_perms))
    p_vals = np.zeros(n_transitions)

    for i in tqdm(np.arange(n_transitions)):
        bystanders = ~np.logical_or(x0_mat[:, i], xf_mat[:, i])
        x = sp.stats.rankdata(state_brain_map[bystanders])

        B_bystanders = B_opt[bystanders, i]
        B_bystanders = sp.stats.rankdata(B_bystanders)

        # observed
        observed[i], _ = sp.stats.pearsonr(x, B_bystanders)

        # null
        for j in np.arange(n_perms):
            B_null = B_opt_network_null[bystanders, i, j]
            r_null[i, j], _ = sp.stats.pearsonr(x, sp.stats.rankdata(B_null))

        p_vals[i] = get_null_p(observed[i], r_null[i, :], abs=True)

    np.save(os.path.join(environment.pipelinedir, 'optimized_weights_r_null_{0}_{1}.npy'.format(which_brain_map, network_null)), r_null)
    np.save(os.path.join(environment.pipelinedir, 'optimized_weights_p_vals_{0}_{1}.npy'.format(which_brain_map, network_null)), p_vals)
    np.save(os.path.join(environment.pipelinedir, 'optimized_weights_observed_{0}_{1}.npy'.format(which_brain_map, network_null)), observed)

# plot
f, ax = plt.subplots(1, 1, figsize=(figsize, figsize))
observed_mean = np.mean(np.abs(observed))
r_null_mean = np.mean(np.abs(r_null), axis=0)
p_val = get_null_p(observed_mean, r_null_mean, abs=True)
my_null_plot(observed=observed_mean, null=r_null_mean, p_val=p_val, xlabel='spatial corr.\n(null network)', ax=ax)
f.savefig(os.path.join(environment.figdir, 'corr(smap,B_opt)_null_{0}'.format(network_null)), dpi=600, bbox_inches='tight', pad_inches=0.01)
plt.close()
