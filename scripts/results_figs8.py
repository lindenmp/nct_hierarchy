# %% import
import sys, os, platform
from src.pipelines import ComputeMinimumControlEnergy
from src.energy import expand_states
from src.plotting import my_bsci_plot

from tqdm import tqdm

# %% import workspace
os.environ["MY_PYTHON_WORKSPACE"] = 'subj_adj'
os.environ["WHICH_BRAIN_MAP"] = 'hist-g2'
from setup_workspace import *

# %% plotting
import seaborn as sns
import matplotlib.pyplot as plt
from src.plotting import set_plotting_params
set_plotting_params(format='svg')
figsize = 1.5

# %% get control energy
c = 1
T = 1
B = DataMatrix(data=np.eye(n_parcels), name='identity')
optimized_weights = np.zeros((n_subs, n_parcels, n_states*n_states))

# set pipelinedir to cluster outputs
environment.pipelinedir = environment.pipelinedir.replace('output_local', 'output_cluster')

for i in tqdm(np.arange(n_subs)):
    file_prefix = '{0}_{1}_'.format(environment.df.index[i], which_brain_map)

    nct_pipeline = ComputeMinimumControlEnergy(environment=environment, A=load_sc.A[:, :, i], states=states, B=B,
                                               control='minimum_fast', c=c, T=T,
                                               file_prefix=file_prefix,
                                               force_rerun=False, save_outputs=True, verbose=False)
    n = 2
    ds = 0.1
    nct_pipeline.run_with_optimized_b(n=n, ds=ds)
    optimized_weights[i, :, :] = nct_pipeline.B_opt[:, :, 1]

# reset pipelinedir to local outputs
environment.pipelinedir = environment.pipelinedir.replace('output_cluster', 'output_local')

# %% correlations
x0_mat, xf_mat = expand_states(states)
n_transitions = x0_mat.shape[1]

r = np.zeros((n_subs, n_transitions))

for i in tqdm(np.arange(n_subs)):
    for j in np.arange(n_transitions):
        bystanders = ~np.logical_or(x0_mat[:, j], xf_mat[:, j])

        dist_from_x0 = sp.stats.rankdata(np.abs(state_brain_map[bystanders] - state_brain_map[x0_mat[:, j]].mean()))
        dist_from_xf = sp.stats.rankdata(np.abs(state_brain_map[bystanders] - state_brain_map[xf_mat[:, j]].mean()))

        optimized_weights_rank = sp.stats.rankdata(optimized_weights[i, bystanders, j])

        # observed from x0
        obs_x0, _ = sp.stats.pearsonr(dist_from_x0, optimized_weights_rank)

        # observed from xf
        obs_xf, _ = sp.stats.pearsonr(dist_from_xf, optimized_weights_rank)

        if np.abs(obs_x0) > np.abs(obs_xf):
            r[i, j] = obs_x0
        else:
            r[i, j] = obs_xf

# %%
r_mean = np.mean(r, axis=1)
# r_mean = np.mean(np.abs(r), axis=1)

f, ax = plt.subplots(1, 1, figsize=(figsize*1.5, figsize*1.5))
my_bsci_plot(dist=r_mean, observed=np.mean(r_mean), xlabel='correlation(cytoarchitecture, optimized weights)', ax=ax)
f.savefig(os.path.join(environment.figdir, 'corr(smap,B_opt_subj)'), dpi=600,
          bbox_inches='tight', pad_inches=0.01)
plt.close()
