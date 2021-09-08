# %% import
import sys, os, platform
from pfactor_gradients.imaging_derivs import DataMatrix
from pfactor_gradients.pipelines import ComputeMinimumControlEnergy
from pfactor_gradients.plotting import my_distpair_plot
from pfactor_gradients.utils import rank_int, get_bootstrap_indices, mean_confidence_interval

import pandas as pd
from tqdm import tqdm

# %% import workspace
from setup_workspace_ave_adj import *

# %% plotting
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
from pfactor_gradients.plotting import set_plotting_params
set_plotting_params(format='svg')
figsize = 1.5

# %% get primary control energy and weights
file_prefix = 'average_adj_n-{0}_cthr-{1}_smap-{2}_'.format(load_average_sc.load_sc.df.shape[0],
                                                            consist_thresh, which_brain_map)

# optimized B weights
B = DataMatrix(data=np.eye(n_parcels), name='identity')
nct_pipeline = ComputeMinimumControlEnergy(environment=environment, A=A, states=states, B=B,
                                           control='minimum_fast', T=1,
                                           file_prefix=file_prefix,
                                           force_rerun=False, save_outputs=True, verbose=True)
n = 2
ds = 0.1
nct_pipeline.run_with_optimized_b(n=n, ds=ds)

e = nct_pipeline.E_opt[:, 0].reshape(n_states, n_states)
e = rank_int(e)  # normalized energy matrix
ed = e.transpose() - e  # energy asymmetry matrix

e_opt = nct_pipeline.E_opt[:, 1].reshape(n_states, n_states)
e_opt = rank_int(e_opt)  # normalized energy matrix
ed_opt = e_opt.transpose() - e_opt  # energy asymmetry matrix

b = nct_pipeline.B_opt[:, :, 0]
b_opt = nct_pipeline.B_opt[:, :, 1]

print(np.mean(ed[indices_upper]))
print(np.mean(ed_opt[indices_upper]))

# %% bootstrap
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
    nct_pipeline = ComputeMinimumControlEnergy(environment=environment, A=load_average_sc_strap.A, states=states, B=B,
                                               control='minimum_fast', T=1,
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

# %% plot
# top-down vs bottom-up
f, ax = plt.subplots(1, 1, figsize=(figsize*0.75, figsize))
df_plot = pd.DataFrame(data=np.vstack((ed_bs, ed_opt_bs)).transpose(),
                       columns=['identity', 'optimized'])
my_distpair_plot(df=df_plot, ylabel='energy delta', ax=ax, test_stat=None, split=True)
f.savefig(os.path.join(environment.figdir, 'e_asym_dists_bootstrap'), dpi=600, bbox_inches='tight',
          pad_inches=0.01)
plt.close()

print(np.round(mean_confidence_interval(data=ed_bs), 4))
print(np.round(mean_confidence_interval(data=ed_opt_bs), 4))
