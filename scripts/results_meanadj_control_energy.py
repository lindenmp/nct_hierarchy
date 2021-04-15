import os
import numpy as np
import scipy as sp
import pandas as pd
from pfactor_gradients.pnc import Environment, Subject
from pfactor_gradients.routines import LoadSC, LoadCT, LoadRLFP, LoadCBF, LoadREHO, LoadALFF, LoadAverageBrainMaps
from pfactor_gradients.imaging_derivs import DataMatrix, DataVector
from pfactor_gradients.pipelines import ComputeGradients
from pfactor_gradients.plotting import my_regplot
from pfactor_gradients.utils import get_pdist_clusters, get_fdr_p
from tqdm import tqdm

# %% Plotting
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='white', context='talk', font_scale=1)
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
compute_gradients = ComputeGradients(environment=environment, Subject=Subject)
compute_gradients.run()

# %% Load sc data
load_sc = LoadSC(environment=environment, Subject=Subject)
load_sc.run()
# refilter environment due to LoadSC excluding on disconnected nodes
environment.df = load_sc.df.copy()

# %% load mean brain maps
loaders_dict = {
    'ct': LoadCT(environment=environment, Subject=Subject),
    # 'rlfp': LoadRLFP(environment=environment, Subject=Subject),
    'cbf': LoadCBF(environment=environment, Subject=Subject),
    'reho': LoadREHO(environment=environment, Subject=Subject),
    'alff': LoadALFF(environment=environment, Subject=Subject)
}

load_average_bms = LoadAverageBrainMaps(loaders_dict=loaders_dict)
load_average_bms.run()

for key in load_average_bms.brain_maps:
    load_average_bms.brain_maps[key].mean_between_states(compute_gradients.grad_bins)

n_states = len(np.unique(compute_gradients.grad_bins))
mask = ~np.eye(n_states, dtype=bool)
indices = np.where(mask)

# %% get minimum control energy
my_list = ['wb',] + list(load_average_bms.brain_maps.keys())
df = pd.DataFrame()

for i in my_list:
    file = 'average_adj_n-775_s-0.06_ns-40-0_c-minimum_fast_T-1_B-{0}_E.npy'.format(i)
    E = np.load(os.path.join(environment.pipelinedir, 'minimum_control_energy', file))
    df[i] = E[indices]

# %% energy distributions
f, ax = plt.subplots(1, 1, figsize=(5, 2.5))
sns.violinplot(data=df, ax=ax, scale='width')
f.savefig(os.path.join(environment.figdir, 'energy_dist.png'), dpi=150, bbox_inches='tight',
          pad_inches=0.1)
plt.close()

for i in my_list:
    for j in my_list:
        if i != j:
            if sp.stats.ttest_rel(df[i], df[j])[1] < 0.05:
                print('t test', i, j, np.round(sp.stats.ttest_rel(df[i], df[j]), 4))
            if sp.stats.ks_2samp(df[i], df[j])[1] < 0.05:
                print('ks test', i, j, np.round(sp.stats.ks_2samp(df[i], df[j]), 4))

# %% Does energy correlate to corresponding brain maps?
for key in load_average_bms.brain_maps:
    f, ax = plt.subplots(1, 1, figsize=(5, 5))
    my_regplot(load_average_bms.brain_maps[key].data_mean[indices], df[key], key, 'Energy ({0})'.format(key), ax)
    f.savefig(os.path.join(environment.figdir, 'corr({0},energy_{0}).png'.format(key)), dpi=150, bbox_inches='tight',
              pad_inches=0.1)
    plt.close()

# %% How does energy diverge from identity when adding a brain map?
for key in load_average_bms.brain_maps:
    f, ax = plt.subplots(1, 1, figsize=(5, 5))
    my_regplot(df['wb'], df[key], 'Energy (wb)', 'Energy ({0})'.format(key), ax)
    f.savefig(os.path.join(environment.figdir, 'corr(energy_wb,energy_{0}).png'.format(key)), dpi=150, bbox_inches='tight',
              pad_inches=0.1)
    plt.close()

# %% Does energy vary over the state transitions according to the neurobiological feature that is used for optimization?
df_corr = pd.DataFrame(index=my_list, columns=my_list)
df_p = pd.DataFrame(index=my_list, columns=my_list)

for i in my_list:
    for j in my_list:
        df_corr.loc[i, j] = sp.stats.spearmanr(df[i], df[j])[0]
        df_p.loc[i, j] = sp.stats.spearmanr(df[i], df[j])[1]
df_corr = df_corr.astype(float)
df_p = df_p.astype(float)

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(df_corr, dtype=bool))
n_tests = np.sum(~mask.flatten())

# p_corr = get_fdr_p(df_p.values.flatten()[~mask.flatten()])
# df_p_corr = df_p.copy()
# df_p_corr[:] = np.nan
# df_p_corr.iloc[~mask] = p_corr
# mask = np.logical_or(mask, df_p_corr > 0.05)

mask = np.logical_or(mask, df_p > (0.05 / n_tests))

f, ax = plt.subplots(1, 1, figsize=(5, 5))
sns.heatmap(df_corr, mask=mask, ax=ax, square=True, center=0, cmap='coolwarm')
f.savefig(os.path.join(environment.figdir, 'corr_energy.png'), dpi=150, bbox_inches='tight',
          pad_inches=0.1)
plt.close()
