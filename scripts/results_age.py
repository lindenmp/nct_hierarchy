import sys, os, platform
from pfactor_gradients.pnc import Environment, Subject
from pfactor_gradients.routines import LoadSC, LoadCT, LoadRLFP, LoadCBF, LoadREHO, LoadALFF,\
    LoadAverageSC, LoadAverageBrainMaps
from pfactor_gradients.pipelines import ComputeGradients, ComputeMinimumControlEnergy
from pfactor_gradients.utils import rank_int, get_fdr_p
from pfactor_gradients.plotting import my_regplot
import numpy as np
import pandas as pd
import scipy as sp
from tqdm import tqdm
from sklearn.linear_model import LinearRegression

# %% Plotting
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='white', context='paper', font_scale=1)
import matplotlib.font_manager as font_manager
fontpath = '/Users/lindenmp/Library/Fonts/PublicSans-Thin.ttf'
prop = font_manager.FontProperties(fname=fontpath)
plt.rcParams['font.family'] = prop.get_name()
plt.rcParams['svg.fonttype'] = 'none'

# %% Setup project environment
computer = 'macbook'
parc = 'schaefer'
n_parcels = 400
sc_edge_weight = 'streamlineCount'
environment = Environment(computer=computer, parc=parc, n_parcels=n_parcels, sc_edge_weight=sc_edge_weight)
environment.make_output_dirs()
environment.load_parc_data()

# %% get clustered gradients
filters = {'healthExcludev2': 0, 't1Exclude': 0,
           'b0ProtocolValidationStatus': 1, 'dti64ProtocolValidationStatus': 1, 'dti64Exclude': 0,
           'psychoactiveMedPsychv2': 0, 'restProtocolValidationStatus': 1, 'restExclude': 0}
environment.load_metadata(filters)
n_bins = int(n_parcels/10)
compute_gradients = ComputeGradients(environment=environment, Subject=Subject, n_bins=n_bins)
compute_gradients.run()

n_states = len(np.unique(compute_gradients.grad_bins))
mask = ~np.eye(n_states, dtype=bool)
indices = np.where(mask)
indices_upper = np.triu_indices(n_states, k=1)
indices_lower = np.tril_indices(n_states, k=-1)

# %% Load sc data
load_sc = LoadSC(environment=environment, Subject=Subject)
load_sc.run()
# refilter environment due to LoadSC excluding on disconnected nodes
environment.df = load_sc.df.copy()
n_subs = environment.df.shape[0]

# %% get control energy
B = 'wb'
n_subsamples = 0
E = np.zeros((compute_gradients.n_states, compute_gradients.n_states, n_subs))

for i in tqdm(np.arange(n_subs)):
    file_prefix = '{0}_'.format(environment.df.index[i])

    nct_pipeline = ComputeMinimumControlEnergy(environment=environment, A=load_sc.A[:, :, i],
                                               states=compute_gradients.grad_bins, n_subsamples=n_subsamples,
                                               control='minimum_fast', T=1, B='wb', file_prefix=file_prefix,
                                               force_rerun=False, save_outputs=True, verbose=False)
    nct_pipeline.run()
    E[:, :, i] = nct_pipeline.E

# normalize energy
for i in tqdm(np.arange(n_states)):
    for j in np.arange(n_states):
        E[i, j, :] = rank_int(E[i, j, :])

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

# %% age correlations
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
f, ax = plt.subplots(1, 1, figsize=(3, 3))
sns.heatmap(e_corr, mask=sig_mask, center=0, square=True, cmap='coolwarm', ax=ax)
plt.subplots_adjust(wspace=.25)
f.savefig(os.path.join(environment.figdir, 'energy_age_corr.png'), dpi=150, bbox_inches='tight',
          pad_inches=0.1)
plt.close()

f, ax = plt.subplots(1, 1, figsize=(3, 3))
my_regplot(x=e_corr[indices_upper], y=e_corr[indices_lower], xlabel='Age (bottom up)', ylabel='Age (top down)', ax=ax)
plt.subplots_adjust(wspace=.25)
f.savefig(os.path.join(environment.figdir, 'energy_age_corr_corr.png'), dpi=150, bbox_inches='tight',
          pad_inches=0.1)
plt.close()
