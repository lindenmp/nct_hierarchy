# %% import
import sys, os, platform
from src.pipelines import ComputeMinimumControlEnergy
from src.utils import rank_int, mean_over_states

from tqdm import tqdm

# %% import workspace
os.environ["MY_PYTHON_WORKSPACE"] = 'ave_adj'
os.environ["WHICH_BRAIN_MAP"] = 'hist-g2'
os.environ["INTRAHEMI"] = "False"
from setup_workspace import *

# %% plotting
import seaborn as sns
import matplotlib.pyplot as plt
from src.plotting import set_plotting_params
set_plotting_params(format='svg')
figsize = 1.5

# %% get control energy

# get hierarchy distance between states
states_distance = mean_over_states(state_brain_map, states)
states_distance = np.abs(states_distance)

file_prefix = 'average_adj_n-{0}_cthr-{1}_smap-{2}_'.format(load_average_sc.load_sc.df.shape[0],
                                                            consist_thresh, which_brain_map)
B = DataMatrix(data=np.eye(n_parcels), name='identity')

c_array = np.array([0.1, 1, 10, 10**2, 10**3, 10**4, 10**5, 10**6])
n_c = len(c_array)

T_array = np.arange(1, 11)
n_T = len(T_array)

ed_mean = np.zeros((n_c, n_T))
ed_dist_corr = np.zeros((n_c, n_T))

for i in tqdm(np.arange(n_c)):
    c = c_array[i]
    if c >= 1:
        c = int(c)
    for j in np.arange(n_T):
        T = T_array[j]

        nct_pipeline = ComputeMinimumControlEnergy(environment=environment, A=A, states=states, B=B,
                                                   control='minimum_fast', c=c, T=T,
                                                   file_prefix=file_prefix,
                                                   force_rerun=False, save_outputs=True, verbose=False)
        nct_pipeline.run()

        # get energy
        e = nct_pipeline.E
        e = rank_int(e)  # normalized energy matrix
        ed = e - e.transpose()  # energy asymmetry matrix

        ed_mean[i, j] = np.mean(ed[indices_upper])  # mean energy asymmetry
        ed_dist_corr[i, j] = sp.stats.spearmanr(states_distance[indices_upper], ed[indices_upper])[0]  # distance corr

#%%
f, ax = plt.subplots(1, 2, figsize=(6, 3))
sns.heatmap(ed_mean, center=0, square=True, cmap='coolwarm', ax=ax[0],
            cbar_kws={"shrink": 0.5, "label": "mean asymmetry"})
ax[0].set_title('mean energy asymmetry')

sns.heatmap(ed_dist_corr, center=0, square=True, cmap='coolwarm', ax=ax[1],
            cbar_kws={"shrink": 0.5, "label": "Spearman's rho"})
ax[1].set_title('hierarchy distance correlation')

for a in ax:
    a.set_xticks(np.arange(n_T)+0.5)
    a.set_xticklabels(T_array)

    a.set_yticks(np.arange(n_c)+0.5)
    a.set_yticklabels(['0.1', '1', '10', '$10^2$', '$10^3$', '$10^4$', '$10^5$', '$10^6$'], rotation=0)

    a.set_xlabel('time horizon, T')
    a.set_ylabel('normalization parameter, c')

    a.tick_params(pad=-2.5)
f.tight_layout()
f.savefig(os.path.join(environment.figdir, 'sensitivity_c_T'), dpi=600, bbox_inches='tight', pad_inches=0.01)
plt.close()
