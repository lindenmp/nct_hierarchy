# %% import
import sys, os, platform
from src.pipelines import ComputeMinimumControlEnergy
from src.plotting import my_bsci_pair_plot
from src.utils import rank_int, get_null_p, get_fdr_p
from src.utils import get_bootstrap_indices

from brainsmash.mapgen.base import Base
from tqdm import tqdm

# %% import workspace
os.environ["MY_PYTHON_WORKSPACE"] = 'ave_adj'
os.environ["WHICH_BRAIN_MAP"] = 'hist-g2'
from setup_workspace import *

# %% plotting
import seaborn as sns
import matplotlib.pyplot as plt
from src.plotting import set_plotting_params
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

# %%
e = E['identity'].copy()
e = rank_int(e)  # normalized energy matrix
ed = e - e.transpose()  # energy asymmetry matrix

e = E['E_opt'].copy()
e = rank_int(e)  # normalized energy matrix
ed_opt = e - e.transpose()  # energy asymmetry matrix

B_opt_matrix = np.zeros((n_states, n_states, n_parcels))
row = 0
col = 0

n_transitions = n_states * n_states
for i in tqdm(np.arange(n_transitions)):
    B_opt_matrix[row, col, :] = B_opt[:, i].copy()
    if col == n_states-1:
        row += 1
        col = 0
    else:
        col += 1

# %% generate surrogates using brainsmash
n_surrogates = 10000
file = 'brainsmash_surrogates_{0}_n{1}.npy'.format(which_brain_map, n_surrogates)
if os.path.exists(os.path.join(environment.pipelinedir, file)) == False:
    D = sp.spatial.distance.pdist(environment.centroids, 'euclidean')
    D = sp.spatial.distance.squareform(D)

    base = Base(x=state_brain_map, D=D, resample=True)
    surrogates = base(n=n_surrogates)

    np.save(os.path.join(environment.pipelinedir, file), surrogates)
else:
    surrogates = np.load(os.path.join(environment.pipelinedir, file))

# %% correlate hierarchical distance with optimized weights
observed = np.zeros((n_states, n_states))
null = np.zeros((n_states, n_states, n_surrogates))
p_vals = np.zeros((n_states, n_states))

for i in tqdm(np.arange(n_states)):
    for j in np.arange(n_states):
        x0 = states == i
        xf = states == j

        bystanders = ~np.logical_or(x0, xf)

        dist_from_x0 = sp.stats.rankdata(np.abs(state_brain_map[bystanders] - state_brain_map[x0].mean()))
        dist_from_xf = sp.stats.rankdata(np.abs(state_brain_map[bystanders] - state_brain_map[xf].mean()))

        optimized_weights = sp.stats.rankdata(B_opt_matrix[i, j, bystanders])

        # observed from x0
        obs_x0, _ = sp.stats.pearsonr(dist_from_x0, optimized_weights)

        # observed from xf
        obs_xf, _ = sp.stats.pearsonr(dist_from_xf, optimized_weights)

        if np.abs(obs_x0) > np.abs(obs_xf):
            use_x0 = True
            observed[i, j] = obs_x0
        else:
            use_x0 = False
            observed[i, j] = obs_xf

        # null
        for k in np.arange(n_surrogates):
            if use_x0:
                dist_surr = np.abs(surrogates[k, bystanders] - surrogates[k, x0].mean())
            else:
                dist_surr = np.abs(surrogates[k, bystanders] - surrogates[k, xf].mean())

            null[i, j, k], _ = sp.stats.pearsonr(dist_surr, optimized_weights)

        p_vals[i, j] = get_null_p(observed[i, j], null[i, j, :], version='standard', abs=True)

# %% get bootstrapped energy asymmetries for both uniform and optimized control weights
n_samples = 10000
ed_bs = np.zeros(n_samples)
ed_opt_bs = np.zeros(n_samples)

bootstrap_indices = get_bootstrap_indices(d_size=n_subs, frac=1, n_samples=n_samples)

# set pipelinedir to cluster outputs
environment.pipelinedir = environment.pipelinedir.replace('output_local', 'output_cluster')

for i in tqdm(np.arange(n_samples)):
    file_prefix = 'average_adj_n-{0}_cthr-{1}_smap-{2}_strap-{3}_'.format(load_average_sc.load_sc.df.shape[0],
                                                                          consist_thresh, which_brain_map, i)

    # load_sc_strap = LoadSC(environment=environment, Subject=Subject)
    # load_sc_strap.df = load_sc.df.iloc[bootstrap_indices[i, :], :].copy()
    # load_sc_strap.A = load_sc.A[:, :, bootstrap_indices[i, :]].copy()
    #
    # load_average_sc_strap = LoadAverageSC(load_sc=load_sc_strap, consist_thresh=consist_thresh, verbose=False)
    # load_average_sc_strap.run()
    #
    # # get bootstrapped energy
    # nct_pipeline = ComputeMinimumControlEnergy(environment=environment, A=load_average_sc_strap.A, states=states, B=B,
    #                                            control='minimum_fast', c=c, T=T,
    #                                            file_prefix=file_prefix,
    #                                            force_rerun=False, save_outputs=False, verbose=False)
    # n = 2
    # ds = 0.1
    # nct_pipeline.run_with_optimized_b(n=n, ds=ds)
    data = np.load(os.path.join(environment.pipelinedir, 'minimum_control_energy',
                                file_prefix + 'ns-20_ctrl-minimum_fast_c-1_T-1_B-optimized-n-2-ds-0.1_E.npy'))

    # e_strap = nct_pipeline.E_opt[:, 0].reshape(n_states, n_states)
    e_strap = data[:, 0].reshape(n_states, n_states)
    e_strap = rank_int(e_strap)  # normalized energy matrix
    ed_strap = e_strap - e_strap.transpose()  # energy asymmetry matrix
    ed_strap = np.mean(ed_strap[indices_upper])
    ed_bs[i] = ed_strap

    # e_opt_strap = nct_pipeline.E_opt[:, 1].reshape(n_states, n_states)
    e_opt_strap = data[:, 1].reshape(n_states, n_states)
    e_opt_strap = rank_int(e_opt_strap)  # normalized energy matrix
    ed_opt_strap = e_opt_strap - e_opt_strap.transpose()  # energy asymmetry matrix
    ed_opt_strap = np.mean(ed_opt_strap[indices_upper])
    ed_opt_bs[i] = ed_opt_strap

# reset pipelinedir to local outputs
environment.pipelinedir = environment.pipelinedir.replace('output_cluster', 'output_local')

# %% Panel B: correlations
f, ax = plt.subplots(1, 1, figsize=(figsize*1.75, figsize*1.75))
p_vals_fdr = get_fdr_p(p_vals)
sig_mask = p_vals_fdr > 0.05
sns.heatmap(observed, mask=sig_mask, center=0, square=True, cmap='coolwarm', ax=ax, cbar_kws={"shrink": 0.80})
ax.set_ylabel("initial states", labelpad=-1)
ax.set_xlabel("target states", labelpad=-1)
ax.set_yticklabels('')
ax.set_xticklabels('')
ax.tick_params(pad=-1)
f.savefig(os.path.join(environment.figdir, 'corr(observed,brainmap).svg'), dpi=600, bbox_inches='tight', pad_inches=0.01)
plt.close()

# %% Panel C: bootstrap
f, ax = plt.subplots(1, 1, figsize=(figsize*2.5, figsize*1.25))
my_bsci_pair_plot(np.abs(ed_bs), np.abs(np.mean(ed[indices_upper])),
                  np.abs(ed_opt_bs), np.abs(np.mean(ed_opt[indices_upper])),
                  xlabel='absolute mean asymmetry (bootstrap)', ax=ax)
f.savefig(os.path.join(environment.figdir, 'e_ed_asym_bootstrap.svg'), dpi=600, bbox_inches='tight',
          pad_inches=0.01)
plt.close()
