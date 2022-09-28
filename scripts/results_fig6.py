# %% import
import sys, os, platform
from src.pipelines import ComputeMinimumControlEnergy
from src.plotting import my_reg_plot
from src.utils import get_fdr_p
from bct.algorithms.physical_connectivity import density_und

from sklearn.linear_model import LinearRegression
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

# %% get subject edge density
A_d = np.zeros(n_subs)
for i in range(n_subs):
    A_d[i], _, _ = density_und(load_sc.A[:, :, i])

environment.df['edge_density'] = A_d

# %% get control energy
c = 1
T = 1
B = DataMatrix(data=np.eye(n_parcels), name='identity')
E = np.zeros((n_states, n_states, n_subs))
# energy_version = 'identity'
energy_version = 'E_opt'
ed_mean = np.load(os.path.join(environment.pipelinedir, 'ed_{0}_{1}.npy'.format(which_brain_map, energy_version)))

# set pipelinedir to cluster outputs
environment.pipelinedir = environment.pipelinedir.replace('output_local', 'output_cluster')

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

# reset pipelinedir to local outputs
environment.pipelinedir = environment.pipelinedir.replace('output_cluster', 'output_local')

# %% correlations
dv = 'ageAtScan1'
y = environment.df.loc[:, dv].values

if np.any(np.isnan(y)):
    missing_data = np.isnan(y)
    print('filter {0} missing subjects...'.format(np.sum(missing_data)))
    # y = y[~missing_data]
    print('imputing missing data...')
    y[np.isnan(y)] = np.nanmedian(y)

# nuisance regression
covs = environment.df.loc[:, ['sex', 'mprage_antsCT_vol_TBV', 'dti64MeanRelRMS', 'edge_density']]
covs['sex'] = covs['sex'] - 1
covs = covs.values

for i in tqdm(np.arange(n_states)):
    for j in np.arange(n_states):
        nuis_reg = LinearRegression()
        nuis_reg.fit(covs, E[i, j, :])
        y_pred = nuis_reg.predict(covs)
        Em = np.mean(E[i, j, :])
        E[i, j, :] = E[i, j, :] - y_pred
        E[i, j, :] += Em

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
y = y/12

# Panel A
f, ax = plt.subplots(1, 1, figsize=(figsize*1.2, figsize*1.2))
cmap = sns.diverging_palette(150, 275, as_cmap=True)
sns.heatmap(e_corr, mask=sig_mask, center=0, square=True, cmap=cmap, ax=ax, vmin=-0.2, vmax=0.2,
            cbar_kws={"shrink": 0.80, "label": "age effects\n(Pearson's r)"})
ax.set_ylabel("initial states", labelpad=-1)
ax.set_xlabel("target states", labelpad=-1)
ax.set_yticklabels('')
ax.set_xticklabels('')
ax.tick_params(pad=-2.5)
f.savefig(os.path.join(environment.figdir, 'corr(age,e)'), dpi=600, bbox_inches='tight', pad_inches=0.01)
plt.close()

r = np.zeros(4,)
p = np.zeros(4,)
r[0], p[0] = sp.stats.pearsonr(y, e_bu_mean - e_td_mean)
r[1], p[1] = sp.stats.pearsonr(e_corr[indices], ed_mean[indices])
r[2], p[2] = sp.stats.pearsonr(y, e_bu_mean)
r[3], p[3] = sp.stats.pearsonr(y, e_td_mean)
print(p, p < (.05/3))
p = get_fdr_p(p)
print(p, p < .05)

# New panel B
f, ax = plt.subplots(1, 1, figsize=(figsize, figsize))
my_reg_plot(x=y, y=e_bu_mean - e_td_mean, xlabel='age (years)', ylabel='energy asymmetry\n(mean)', ax=ax, annotate=(r[0], p[0]))
f.savefig(os.path.join(environment.figdir, 'corr(age,e_asym_mean)'), dpi=600, bbox_inches='tight', pad_inches=0.01)
plt.close()

# Panel C
f, ax = plt.subplots(1, 1, figsize=(figsize, figsize))
my_reg_plot(x=e_corr[indices], y=ed_mean[indices], xlabel='age effects', ylabel='energy asymmetry', ax=ax, annotate=(r[1], p[1]))
f.savefig(os.path.join(environment.figdir, 'corr(age_effects,ed)'), dpi=600, bbox_inches='tight', pad_inches=0.01)
plt.close()

# supp figure
# Panel A
f, ax = plt.subplots(1, 1, figsize=(figsize, figsize))
my_reg_plot(x=y, y=e_bu_mean, xlabel='age (years)', ylabel='bottom-up energy\n(mean)', ax=ax, annotate=(r[2], p[2]))
f.savefig(os.path.join(environment.figdir, 'corr(age,e_bu_mean)'), dpi=600, bbox_inches='tight', pad_inches=0.01)
plt.close()

# Panel B
f, ax = plt.subplots(1, 1, figsize=(figsize, figsize))
my_reg_plot(x=y, y=e_td_mean, xlabel='age (years)', ylabel='top-down energy\n(mean)', ax=ax, annotate=(r[3], p[3]))
f.savefig(os.path.join(environment.figdir, 'corr(age,e_td_mean)'), dpi=600, bbox_inches='tight', pad_inches=0.01)
plt.close()
