# %% import
import sys, os, platform
from pfactor_gradients.pipelines import ComputeMinimumControlEnergy
from pfactor_gradients.imaging_derivs import DataMatrix
from pfactor_gradients.utils import mean_confidence_interval
from pfactor_gradients.energy import expand_states

import scipy as sp
from tqdm import tqdm

# %% import workspace
os.environ["MY_PYTHON_WORKSPACE"] = 'subj_adj'
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
c = 1
T = 1
B = DataMatrix(data=np.eye(n_parcels), name='identity')
optimized_weights = np.zeros((n_subs, n_parcels, n_states*n_states))

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

# %% correlations
x0_mat, xf_mat = expand_states(states)
n_transitions = x0_mat.shape[1]

r = np.zeros((n_subs, n_transitions))

for i in tqdm(np.arange(n_subs)):
    for j in np.arange(n_transitions):
        bystanders = ~np.logical_or(x0_mat[:, j], xf_mat[:, j])
        r[i, j] = sp.stats.spearmanr(state_brain_map[bystanders], optimized_weights[i, bystanders, j])[0]

# %%
color_blue = sns.color_palette("Set1")[1]

r_mean = np.mean(np.abs(r), axis=1)

f, ax = plt.subplots(1, 1, figsize=(figsize*1.5, figsize*1.5))
sns.histplot(x=r_mean, ax=ax, color='gray')
ax.axvline(x=np.mean(r_mean), ymax=1, clip_on=False, linewidth=1, color=color_blue)
ax.grid(False)
sns.despine(right=True, top=True, ax=ax)
ax.tick_params(pad=-2.5)
ax.set_xlabel('correlation(hierarchy,optimized weights)', labelpad=-0.5)
ax.set_ylabel('counts', labelpad=-0.5)

textstr = 'mean = {:.2f}'.format(np.mean(r_mean))
ax.text(np.mean(r_mean), ax.get_ylim()[1], textstr, fontsize=8,
        horizontalalignment='left', verticalalignment='top', rotation=270, c=color_blue)

f.savefig(os.path.join(environment.figdir, 'corr(smap,B_opt_subj)'), dpi=600, bbox_inches='tight',
          pad_inches=0.01)
plt.close()

print(np.round(mean_confidence_interval(data=r_mean), 4))
