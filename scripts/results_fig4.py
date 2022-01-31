# %% import
import sys, os, platform
from pfactor_gradients.imaging_derivs import DataMatrix
from pfactor_gradients.pipelines import ComputeMinimumControlEnergy
from pfactor_gradients.plotting import my_reg_plot
from pfactor_gradients.utils import rank_int

import scipy as sp

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

# %% Panel B: timescales delta
timescales_delta = np.zeros((n_states, n_states))
for i in np.arange(n_states):
    for j in np.arange(n_states):
        timescales_delta[i, j] = np.nanmean(brain_maps['tau'].data[states == i]) - \
                                 np.nanmean(brain_maps['tau'].data[states == j])
# sign of this timescales_delta matrix is currently unintuitive.
#   if state_i = 0.3 and state_j = 0.5, then 0.3-0.5=-0.2.
#   likewise, if state_i = 0.5 and state_j = 0.3, then 0.5-0.3=0.2.
# thus, an increase in rlfp over states is encoded by a negative number and a decrease is encoded by a positive
# number. Not good! sign flip for intuition
timescales_delta = timescales_delta * -1
# now, negative sign represents decreasing over states and positive sign represents  increasing over states.

# matrix
f, ax = plt.subplots(1, 1, figsize=(figsize * 1.2, figsize * 1.2))
sns.heatmap(timescales_delta, center=0, vmin=np.floor(np.min(timescales_delta)), vmax=np.ceil(np.max(timescales_delta)),
            square=True, cmap='coolwarm', ax=ax, cbar_kws={"shrink": 0.60})
ax.set_ylabel("initial states", labelpad=-1)
ax.set_xlabel("target states", labelpad=-1)
ax.set_yticklabels('')
ax.set_xticklabels('')
ax.tick_params(pad=-2.5)
f.savefig(os.path.join(environment.figdir, 'timescales_delta_matrix'), dpi=600, bbox_inches='tight', pad_inches=0.01)
plt.close()

# energy asymmetry vs timescales delta
f, ax = plt.subplots(1, 1, figsize=(figsize, figsize))
my_reg_plot(x=timescales_delta[indices_upper], y=ed[indices_upper],
            xlabel='intrinsic timescales (delta)', ylabel='energy asymmetry',
            ax=ax, annotate='both')
ax.set_ylim([-6, 4])
plt.subplots_adjust(wspace=.25)
f.savefig(os.path.join(environment.figdir, 'ed_timescales_{0}'.format(B)), dpi=600,
          bbox_inches='tight', pad_inches=0.1)
plt.close()

# %% Figure S2: effective connectivity
# load dcm outputs
try:
    file = 'dcm_{0}_ns-{1}_A.mat'.format(which_brain_map, n_states)
    dcm = sp.io.loadmat(os.path.join(environment.pipelinedir, 'spdcm', file))
    ec = dcm['A']
    ec = np.abs(ec)
    ec = rank_int(ec)
    ecd = ec - ec.transpose()

    # effective connectivity matrix
    f, ax = plt.subplots(1, 2, figsize=(figsize * 2.4, figsize * 1.2))
    sns.heatmap(ec, center=0, vmin=np.floor(np.min(ec)), vmax=np.ceil(np.max(ec)),
                square=True, cmap='coolwarm', ax=ax[0], cbar_kws={"shrink": 0.60})
    sns.heatmap(ecd, center=0, vmin=np.floor(np.min(ecd)), vmax=np.ceil(np.max(ecd)),
                square=True, cmap='coolwarm', ax=ax[1], cbar_kws={"shrink": 0.60})
    for i in [0, 1]:
        ax[i].set_ylabel("initial states", labelpad=-1)
        ax[i].set_xlabel("target states", labelpad=-1)
        ax[i].set_yticklabels('')
        ax[i].set_xticklabels('')
        ax[i].tick_params(pad=-2.5)
    f.savefig(os.path.join(environment.figdir, 'ec_matrix'), dpi=600, bbox_inches='tight', pad_inches=0.01)
    plt.close()

    # energy asymmetry vs effective connectivity asymmetry
    f, ax = plt.subplots(1, 1, figsize=(figsize, figsize))
    my_reg_plot(x=ecd[indices_upper], y=ed[indices_upper],
                xlabel='effective connectivity (delta)', ylabel='energy asymmetry',
                ax=ax, annotate='both')
    plt.subplots_adjust(wspace=.25)
    ax.set_xlim([-4, 4])
    ax.set_ylim([-5.25, 5.25])
    f.savefig(os.path.join(environment.figdir, 'ed_ecd_{0}'.format(B)), dpi=600,
              bbox_inches='tight', pad_inches=0.1)
    plt.close()
except FileNotFoundError:
    print('Requisite files not found...')