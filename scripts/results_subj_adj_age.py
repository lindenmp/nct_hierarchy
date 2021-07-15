# %%
import sys, os, platform

if platform.system() == 'Linux':
    sys.path.extend(['/cbica/home/parkesl/research_projects/pfactor_gradients'])
from pfactor_gradients.routines import LoadCT, LoadSA
from pfactor_gradients.pipelines import ComputeMinimumControlEnergy
from pfactor_gradients.imaging_derivs import DataMatrix
from pfactor_gradients.plotting import my_reg_plot
from pfactor_gradients.utils import rank_int, get_fdr_p

import scipy as sp
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

# %% import workspace
from setup_workspace_subj_adj import *

# %% plotting
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
from pfactor_gradients.plotting import set_plotting_params
set_plotting_params(format='png')
figsize = 1.5

# %% Load sc data
load_sc = LoadSC(environment=environment, Subject=Subject)
load_sc.run()
# refilter environment due to LoadSC excluding on disconnected nodes
environment.df = load_sc.df.copy()
n_subs = environment.df.shape[0]

# %% load mean brain maps
loaders_dict = {
    'ct': LoadCT(environment=environment, Subject=Subject),
    'sa': LoadSA(environment=environment, Subject=Subject)
}

for key in loaders_dict:
    loaders_dict[key].run()

# refilter environment due to some missing FS subjects
# environment.df = loaders_dict['ct'].df.copy()
# n_subs = environment.df.shape[0]

# %% get control energy
T = 1
B = DataMatrix(data=np.eye(n_parcels), name='identity')
E = np.zeros((n_states, n_states, n_subs))
E_opt = np.zeros((n_states, n_states, n_subs))
ed_mean = np.load(os.path.join(environment.pipelinedir, 'ed_{0}_{1}.npy'.format(which_brain_map, B.name)))

for i in tqdm(np.arange(n_subs)):
    file_prefix = '{0}_{1}_'.format(environment.df.index[i], which_brain_map)

    nct_pipeline = ComputeMinimumControlEnergy(environment=environment, A=load_sc.A[:, :, i], states=states, B=B,
                                               control='minimum_fast', T=T,
                                               file_prefix=file_prefix,
                                               force_rerun=False, save_outputs=True, verbose=False)
    nct_pipeline.run()
    E[:, :, i] = nct_pipeline.E

    nct_pipeline.run_with_optimized_b()
    E_opt[:, :, i] = nct_pipeline.E_opt

# %% normalize energy over subjects
for i in tqdm(np.arange(n_states)):
    for j in np.arange(n_states):
        E[i, j, :] = rank_int(E[i, j, :])

# %% nuisance regression
covs = environment.df.loc[:, ['sex', 'mprage_antsCT_vol_TBV', 'dti64MeanRelRMS']]
covs['sex'] = covs['sex'] - 1
covs['mprage_antsCT_vol_TBV'] = rank_int(covs['mprage_antsCT_vol_TBV'])
covs['dti64MeanRelRMS'] = rank_int(covs['dti64MeanRelRMS'])
covs = covs.values

for i in tqdm(np.arange(n_states)):
    for j in np.arange(n_states):
        nuis_reg = LinearRegression()
        nuis_reg.fit(covs, E[i, j, :])
        y_pred = nuis_reg.predict(covs)
        E[i, j, :] = E[i, j, :] - y_pred

# mean bottom-up and top-down separately
e_bu_mean = np.zeros(n_subs)
e_td_mean = np.zeros(n_subs)

for i in tqdm(np.arange(n_subs)):
    e_bu_mean[i] = np.mean(E[:, :, i][indices_upper])
    e_td_mean[i] = np.mean(E[:, :, i][indices_lower])

# age correlations
e_corr = np.zeros((n_states, n_states))
e_corr_p = np.zeros((n_states, n_states))

for i in np.arange(n_states):
    for j in np.arange(n_states):
        e_corr[i, j], e_corr_p[i, j] = sp.stats.pearsonr(environment.df.loc[:, 'ageAtScan1'], E[i, j, :])

e_corr_p = get_fdr_p(e_corr_p)
sig_mask = e_corr_p > 0.05
sig_mask[np.isnan(e_corr_p)] = True
print(np.sum(sig_mask == False))

# %% plots
figsize = 1.5

f, ax = plt.subplots(1, 1, figsize=(figsize, figsize))
my_reg_plot(x=environment.df.loc[:, 'ageAtScan1']/12, y=e_bu_mean,
           xlabel='age (years)', ylabel='bottom-up energy\n(mean)', ax=ax, annotate='both')
f.savefig(os.path.join(environment.figdir, 'corr(age,e_{0}_bu)'.format(B.name)), dpi=600, bbox_inches='tight',
          pad_inches=0.01)
plt.close()

f, ax = plt.subplots(1, 1, figsize=(figsize, figsize))
my_reg_plot(x=environment.df.loc[:, 'ageAtScan1']/12, y=e_td_mean,
           xlabel='age (years)', ylabel='top-down energy\n(mean)', ax=ax, annotate='both')
f.savefig(os.path.join(environment.figdir, 'corr(age,e_{0}_td)'.format(B.name)), dpi=600, bbox_inches='tight',
          pad_inches=0.01)
plt.close()

f, ax = plt.subplots(1, 1, figsize=(figsize*1.2, figsize*1.2))
cmap = sns.diverging_palette(150, 275, as_cmap=True)
sns.heatmap(e_corr, mask=sig_mask, center=0, square=True, cmap=cmap, ax=ax,
            cbar_kws={"shrink": 0.80, "label": "age effects\n(Pearson's r)"})
            # cbar_kws={"shrink": 0.80})
# ax.set_title("age effects\n(Pearson's $\mathit{r}$)")
ax.set_ylabel("initial states", labelpad=-1)
ax.set_xlabel("target states", labelpad=-1)
ax.set_yticklabels('')
ax.set_xticklabels('')
ax.tick_params(pad=-2.5)
f.savefig(os.path.join(environment.figdir, 'corr(age,e_{0}).svg'.format(B.name)), dpi=600, bbox_inches='tight',
          pad_inches=0.01)
plt.close()

f, ax = plt.subplots(1, 1, figsize=(figsize, figsize))
my_reg_plot(x=e_corr[indices_upper], y=ed_mean[indices_upper],
           xlabel='age effects\n(bottom-up energy)', ylabel='energy asymmetry', ax=ax, annotate='both')
f.savefig(os.path.join(environment.figdir, 'corr(corr(e_{0}_bu,ed))'.format(B.name)), dpi=600, bbox_inches='tight',
          pad_inches=0.01)
plt.close()

f, ax = plt.subplots(1, 1, figsize=(figsize, figsize))
my_reg_plot(x=e_corr[indices_lower], y=ed_mean[indices_lower],
           xlabel='age effects\n(top-down energy)', ylabel='energy asymmetry', ax=ax, annotate='both')
f.savefig(os.path.join(environment.figdir, 'corr(corr(e_{0}_td,ed))'.format(B.name)), dpi=600, bbox_inches='tight',
          pad_inches=0.01)
plt.close()

f, ax = plt.subplots(1, 1, figsize=(figsize, figsize))
my_reg_plot(x=e_corr[indices_upper], y=e_corr[indices_lower],
           xlabel='age effects\n(bottom-up energy)', ylabel='age effects\n(top-down energy)', ax=ax, annotate='both')
f.savefig(os.path.join(environment.figdir, 'corr(corr(e_{0},age))'.format(B.name)), dpi=600, bbox_inches='tight',
          pad_inches=0.01)
plt.close()
