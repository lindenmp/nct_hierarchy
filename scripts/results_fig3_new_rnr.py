# %% import
import sys, os, platform
from src.imaging_derivs import DataMatrix
from src.pipelines import ComputeMinimumControlEnergy
from src.plotting import my_reg_plot, my_distpair_plot, my_null_plot
from src.utils import rank_int, get_null_p, mean_over_states

from bct.algorithms.distance import distance_wei_floyd
from src.communicability import matching_index, cumulative_transitivity_differences, path_transitivity
from oct2py import octave
sys.path.append('usr/local/bin/octave')  # octave install path
octave.addpath('/Users/lindenmp/Google-Drive-Penn/work/research_projects/nct_hierarchy/matlab_functions/bct')  # path to BCT matlab functions

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

# %%
L = 1/A
# Caio's normalization length transform. Results ~identical
# L = -np.log10(np.divide(A, (np.max(A) + np.min(A[A > 0]))))
L[np.isinf(L)] = 0

D, hops, Pmat = distance_wei_floyd(L, transform=None)
states_hops = mean_over_states(hops, states)

# get diffuision efficiency
_, de = octave.diffusion_efficiency(A, nout=2)
ded = de - de.transpose()
ded_states = mean_over_states(ded, states)

# get search info
si = octave.search_information(A, L)
sid = si - si.transpose()
sid_states = mean_over_states(sid, states)

# get path transitivity
pt = path_transitivity(A)
m = matching_index(A)
ptcd = cumulative_transitivity_differences(m, hops, Pmat)
ptcd_states = mean_over_states(ptcd, states)

# %% plots

# heat maps
f, ax = plt.subplots(1, 1, figsize=(figsize, figsize))
sns.heatmap(ded_states, square=True, ax=ax, cbar_kws={"shrink": 0.80})
ax.tick_params(pad=-2.5)
f.savefig(os.path.join(environment.figdir, 'ded_states'), dpi=600, bbox_inches='tight', pad_inches=0.01)
plt.close()

f, ax = plt.subplots(1, 1, figsize=(figsize, figsize))
sns.heatmap(sid_states, square=True, ax=ax, cbar_kws={"shrink": 0.80})
ax.tick_params(pad=-2.5)
f.savefig(os.path.join(environment.figdir, 'sid_states'), dpi=600, bbox_inches='tight', pad_inches=0.01)
plt.close()

f, ax = plt.subplots(1, 1, figsize=(figsize, figsize))
sns.heatmap(ptcd_states, square=True, ax=ax, cbar_kws={"shrink": 0.80})
ax.tick_params(pad=-2.5)
f.savefig(os.path.join(environment.figdir, 'ptcd_states'), dpi=600, bbox_inches='tight', pad_inches=0.01)
plt.close()

# energy correlations
# panel D
f, ax = plt.subplots(1, 1, figsize=(figsize, figsize))
my_reg_plot(ded_states[indices_upper], ed[indices_upper],
                        'diffusion efficiency asymmetry', 'energy asymmetry', ax, annotate='pearson')
f.savefig(os.path.join(environment.figdir, 'corr(ded_states,e_asym)'), dpi=600, bbox_inches='tight',
          pad_inches=0.01)
plt.close()

# panel E
f, ax = plt.subplots(1, 1, figsize=(figsize, figsize))
my_reg_plot(sid_states[indices_upper], ed[indices_upper],
                        'search information asymmetry', 'energy asymmetry', ax, annotate='pearson')
f.savefig(os.path.join(environment.figdir, 'corr(sid_states,e_asym)'), dpi=600, bbox_inches='tight',
          pad_inches=0.01)
plt.close()

# panel F
f, ax = plt.subplots(1, 1, figsize=(figsize, figsize))
my_reg_plot(ptcd_states[indices_upper], ed[indices_upper],
                        'path transitivity asymmetry', 'energy asymmetry', ax, annotate='pearson')
f.savefig(os.path.join(environment.figdir, 'corr(ptcd_states,e_asym)'), dpi=600, bbox_inches='tight',
          pad_inches=0.01)
plt.close()

# %% correlation with brain map
x = [state_brain_map, ]
x_name = ['S-F axis', ]

y = [de, si, pt]
y_name = ['diffusion efficiency\n(mean)', 'search information\n(mean)', 'path transitivity\n(mean)']

for i in np.arange(len(x)):
    for j in np.arange(len(y)):
        f, ax = plt.subplots(1, 1, figsize=(figsize, figsize))
        my_reg_plot(x[i], np.nanmean(y[j], axis=0), x_name[i], y_name[j], ax, annotate='pearson')
        f.savefig(os.path.join(environment.figdir, 'corr({0},{1})'.format(x_name[i][:3], y_name[j][:1])), dpi=600, bbox_inches='tight',
                  pad_inches=0.01)
        plt.close()

y = [ded, sid, ptcd]
y_name = ['diffusion efficiency delta\n(mean)', 'search information delta\n(mean)', 'path transitivity delta\n(mean)']

for i in np.arange(len(x)):
    for j in np.arange(len(y)):
        f, ax = plt.subplots(1, 1, figsize=(figsize, figsize))
        my_reg_plot(x[i], np.nanmean(y[j], axis=0), x_name[i], y_name[j], ax, annotate='pearson')
        f.savefig(os.path.join(environment.figdir, 'corr({0},{1}d)'.format(x_name[i][:3], y_name[j][:1])), dpi=600, bbox_inches='tight',
                  pad_inches=0.01)
        plt.close()
