# %%
import sys, os, platform

if platform.system() == 'Linux':
    sys.path.extend(['/cbica/home/parkesl/research_projects/pfactor_gradients'])
from pfactor_gradients.pipelines import ComputeMinimumControlEnergy
from pfactor_gradients.imaging_derivs import DataMatrix
from pfactor_gradients.plotting import my_reg_plot
from pfactor_gradients.utils import rank_int, get_fdr_p

import scipy as sp
from sklearn.linear_model import LinearRegression
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
E = np.zeros((n_states, n_states, n_subs))
# energy_version = 'identity'
energy_version = 'E_opt'
ed_mean = np.load(os.path.join(environment.pipelinedir, 'ed_{0}_{1}.npy'.format(which_brain_map, energy_version)))

for i in tqdm(np.arange(n_subs)):
    file_prefix = '{0}_{1}_'.format(environment.df.index[i], which_brain_map)

    nct_pipeline = ComputeMinimumControlEnergy(environment=environment, A=load_sc.A[:, :, i], states=states, B=B,
                                               control='minimum_fast', c=c, T=T,
                                               file_prefix=file_prefix,
                                               force_rerun=False, save_outputs=True, verbose=False)
    if energy_version == 'identity':
        nct_pipeline.run()
        E[:, :, i] = nct_pipeline.E
    elif energy_version == 'E_opt':
        n = 2
        ds = 0.1
        nct_pipeline.run_with_optimized_b(n=n, ds=ds)
        E[:, :, i] = nct_pipeline.E_opt[:, 1].reshape(n_states, n_states)


# %% normalize energy over subjects
for i in tqdm(np.arange(n_states)):
    for j in np.arange(n_states):
        E[i, j, :] = rank_int(E[i, j, :])

# %% correlations
dv = 'ageAtScan1'
# dv = 'F1_Exec_Comp_Res_Accuracy'

y = environment.df.loc[:, dv].values

if np.any(np.isnan(y)):
    missing_data = np.isnan(y)
    print('filter {0} missing subjects...'.format(np.sum(missing_data)))
    # y = y[~missing_data]
    print('imputing missing data...')
    y[np.isnan(y)] = np.nanmedian(y)

# nuisance regression
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
        e_corr[i, j], e_corr_p[i, j] = sp.stats.pearsonr(y, E[i, j, :])

e_corr_p = get_fdr_p(e_corr_p)
sig_mask = e_corr_p > 0.05
sig_mask[np.isnan(e_corr_p)] = True
print(np.sum(sig_mask == False))

# %% plots
figsize = 1.5

if dv == 'ageAtScan1':
    y = y/12

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
f.savefig(os.path.join(environment.figdir, 'corr(age,e_{0})'.format(energy_version)), dpi=600, bbox_inches='tight',
          pad_inches=0.01)
plt.close()

f, ax = plt.subplots(1, 1, figsize=(figsize, figsize))
my_reg_plot(x=y, y=e_bu_mean,
           xlabel='age (years)', ylabel='bottom-up energy\n(mean)', ax=ax, annotate='pearson')
f.savefig(os.path.join(environment.figdir, 'corr(age,e_{0}_bu)'.format(energy_version)), dpi=600, bbox_inches='tight',
          pad_inches=0.01)
plt.close()

f, ax = plt.subplots(1, 1, figsize=(figsize, figsize))
my_reg_plot(x=y, y=e_td_mean,
           xlabel='age (years)', ylabel='top-down energy\n(mean)', ax=ax, annotate='pearson')
f.savefig(os.path.join(environment.figdir, 'corr(age,e_{0}_td)'.format(energy_version)), dpi=600, bbox_inches='tight',
          pad_inches=0.01)
plt.close()

f, ax = plt.subplots(1, 1, figsize=(figsize, figsize))
my_reg_plot(x=e_corr[indices], y=ed_mean[indices],
           xlabel='age effects\n', ylabel='energy asymmetry', ax=ax, annotate='both')
f.savefig(os.path.join(environment.figdir, 'corr(corr(e_{0},ed))'.format(energy_version)), dpi=600, bbox_inches='tight',
          pad_inches=0.01)
plt.close()

f, ax = plt.subplots(1, 1, figsize=(figsize, figsize))
my_reg_plot(x=y, y=e_bu_mean-e_td_mean,
           xlabel='age (years)', ylabel='energy asymmetry diff\n(mean)', ax=ax, annotate='pearson')
f.savefig(os.path.join(environment.figdir, 'corr(age,e_{0}_td-bu)'.format(energy_version)), dpi=600, bbox_inches='tight',
          pad_inches=0.01)
plt.close()

# %%

# f, ax = plt.subplots(1, 1, figsize=(figsize, figsize))
# my_reg_plot(x=e_corr[indices_upper], y=ed_mean[indices_upper],
#            xlabel='age effects\n(bottom-up energy)', ylabel='energy asymmetry', ax=ax, annotate='both')
# f.savefig(os.path.join(environment.figdir, 'corr(corr(e_{0}_bu,ed))'.format(energy_version)), dpi=600, bbox_inches='tight',
#           pad_inches=0.01)
# plt.close()
#
# f, ax = plt.subplots(1, 1, figsize=(figsize, figsize))
# my_reg_plot(x=e_corr[indices_lower], y=ed_mean[indices_lower],
#            xlabel='age effects\n(top-down energy)', ylabel='energy asymmetry', ax=ax, annotate='both')
# f.savefig(os.path.join(environment.figdir, 'corr(corr(e_{0}_td,ed))'.format(energy_version)), dpi=600, bbox_inches='tight',
#           pad_inches=0.01)
# plt.close()

# f, ax = plt.subplots(1, 1, figsize=(figsize, figsize))
# my_reg_plot(x=e_corr[indices_upper], y=e_corr[indices_lower],
#            xlabel='age effects\n(bottom-up energy)', ylabel='age effects\n(top-down energy)', ax=ax, annotate='both')
# f.savefig(os.path.join(environment.figdir, 'corr(corr(e_{0},age))'.format(energy_version)), dpi=600, bbox_inches='tight',
#           pad_inches=0.01)
# plt.close()
