# %% import
import sys, os, platform

from src.plotting import my_bsci_plot
from tqdm import tqdm
from bct.algorithms.physical_connectivity import density_und

# %% import workspace
os.environ["MY_PYTHON_WORKSPACE"] = 'ave_adj'
os.environ["WHICH_BRAIN_MAP"] = 'hist-g2'
# os.environ["WHICH_BRAIN_MAP"] = 'micro-g1'
# os.environ["WHICH_BRAIN_MAP"] = 'func-g1'
# os.environ["WHICH_BRAIN_MAP"] = 'myelin'
from setup_workspace import *

# %% plotting
import seaborn as sns
import matplotlib.pyplot as plt
from src.plotting import set_plotting_params
set_plotting_params(format='svg')
figsize = 1.5

# %% load bootstrapped A
B = DataMatrix(data=np.eye(n_parcels), name='identity')
c = 1
T = 1
frac = 0.5

A_file = 'average_adj_n-{0}_cthr-{1}_smap-{2}_bootstrapped_frac{3}_Am'.format(load_average_sc.load_sc.df.shape[0],
                                                                              consist_thresh, which_brain_map,
                                                                              str(frac).replace('.', ''))

print('loading bootstrapped A')
Am_bs = np.load(os.path.join(environment.pipelinedir, 'minimum_control_energy', A_file+'.npy'))
n_samples = Am_bs.shape[2]

# %%
A_mean = np.zeros(n_samples)
A_median = np.zeros(n_samples)
A_var = np.zeros(n_samples)
A_std = np.zeros(n_samples)
A_sum = np.zeros(n_samples)
A_d = np.zeros(n_samples)

for i in tqdm(np.arange(n_samples)):
    non_zero_idx = Am_bs[:, :, i] != 0
    A_mean[i] = np.mean(Am_bs[:, :, i][non_zero_idx])
    A_median[i] = np.median(Am_bs[:, :, i][non_zero_idx])
    A_var[i] = np.var(Am_bs[:, :, i][non_zero_idx])
    A_std[i] = np.std(Am_bs[:, :, i][non_zero_idx])

    A_sum[i] = np.sum(Am_bs[:, :, i])
    A_d[i], _, _ = density_und(Am_bs[:, :, i])

plot_list = [A_mean, A_median, A_std, A_sum]
non_zero_idx = A != 0
plot_list2 = [np.mean(A[non_zero_idx]), np.median(A[non_zero_idx]), np.std(A[non_zero_idx]), np.sum(A[non_zero_idx]) ]
plot_labels = ['mean of edge weights', 'median of edge weights', 'std of edge weights', 'sum of edge weights']

for i in np.arange(len(plot_list)):
    f, ax = plt.subplots(1, 1, figsize=(figsize, figsize))
    my_bsci_plot(dist=plot_list[i], observed=plot_list2[i], xlabel=plot_labels[i], ax=ax)
    f.savefig(os.path.join(environment.figdir, plot_labels[i]), dpi=600, bbox_inches='tight', pad_inches=0.01)
    plt.close()
