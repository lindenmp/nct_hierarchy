import os
import numpy as np
import scipy as sp
from pfactor_gradients.pnc import Environment, Subject
from pfactor_gradients.routines import LoadSC, LoadCT, LoadRLFP, LoadCBF, LoadREHO, LoadALFF
from pfactor_gradients.imaging_derivs import DataVector
from pfactor_gradients.pipelines import ComputeGradients
from pfactor_gradients.plotting import my_regplot
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

grad = DataVector(data=compute_gradients.gradients[:, 0], name='grad_1')
# grad.rankdata()
# grad.rescale_unit_interval()
grad.brain_surface_plot(environment)

grad = DataVector(data=compute_gradients.gradients[:, 1], name='grad_2')
# grad.rankdata()
# grad.rescale_unit_interval()
grad.brain_surface_plot(environment)

# %% Load sc data
load_sc = LoadSC(environment=environment, Subject=Subject)
load_sc.run()
# refilter environment due to LoadSC excluding on disconnected nodes
environment.df = load_sc.df.copy()

# %% load ct data
load_ct = LoadCT(environment=environment, Subject=Subject)
load_ct.run()
ct = DataVector(data=np.nanmean(load_ct.ct, axis=0), name='ct')
ct.rankdata()
ct.rescale_unit_interval()
ct.brain_surface_plot(environment)

# %% load rlfp data
load_rlfp = LoadRLFP(environment=environment, Subject=Subject)
load_rlfp.run()
rlfp = DataVector(data=np.nanmean(load_rlfp.rlfp, axis=0), name='rlfp')
rlfp.rankdata()
rlfp.rescale_unit_interval()
rlfp.brain_surface_plot(environment)

# %% load cbf data
load_cbf = LoadCBF(environment=environment, Subject=Subject)
load_cbf.run()
cbf = DataVector(data=np.nanmean(load_cbf.cbf, axis=0), name='cbf')
cbf.rankdata()
cbf.rescale_unit_interval()
cbf.brain_surface_plot(environment)

# %% load reho data
load_reho = LoadREHO(environment=environment, Subject=Subject)
load_reho.run()
reho = DataVector(data=np.nanmean(load_reho.reho, axis=0), name='reho')
reho.rankdata()
reho.rescale_unit_interval()
reho.brain_surface_plot(environment)

# %% load alff data
load_alff = LoadALFF(environment=environment, Subject=Subject)
load_alff.run()
alff = DataVector(data=np.nanmean(load_alff.alff, axis=0), name='alff')
alff.rankdata()
alff.rescale_unit_interval()
alff.brain_surface_plot(environment)

# %% spatial correlations with grad
B_list = [ct, rlfp, cbf, reho, alff]
for B in B_list:
    f, ax = plt.subplots(1, 2, figsize=(10, 5))
    my_regplot(compute_gradients.gradients[:, 0], B.data, 'Gradient 1', B.name, ax[0])
    my_regplot(compute_gradients.gradients[:, 1], B.data, 'Gradient 2', B.name, ax[1])
    f.savefig(os.path.join(environment.figdir, 'corr(grads,{0}).png'.format(B.name)), dpi=150, bbox_inches='tight',
              pad_inches=0.1)
    plt.close()

# %% correlations between gradient and subject maps
n_subs = environment.df.shape[0]
grad_ct_corr = np.zeros(n_subs)
grad_rlfp_corr = np.zeros(n_subs)
grad_cbf_corr = np.zeros(n_subs)
grad_reho_corr = np.zeros(n_subs)
grad_alff_corr = np.zeros(n_subs)

for i in tqdm(np.arange(environment.df.shape[0])):
    grad_ct_corr[i] = sp.stats.spearmanr(compute_gradients.gradients[:, 0], load_ct.ct[i, :])[0]
    grad_rlfp_corr[i] = sp.stats.spearmanr(compute_gradients.gradients[:, 0], load_rlfp.rlfp[i, :])[0]
    grad_cbf_corr[i] = sp.stats.spearmanr(compute_gradients.gradients[:, 0], load_cbf.cbf[i, :])[0]
    grad_reho_corr[i] = sp.stats.spearmanr(compute_gradients.gradients[:, 0], load_reho.reho[i, :])[0]
    grad_alff_corr[i] = sp.stats.spearmanr(compute_gradients.gradients[:, 0], load_alff.alff[i, :])[0]

f, ax = plt.subplots(1, 5, figsize=(20, 4))
sns.histplot(x=grad_ct_corr, ax=ax[0])
ax[0].set_title('corr(grad,ct)')
sns.histplot(x=grad_rlfp_corr, ax=ax[1])
ax[1].set_title('corr(grad,rlfp)')
sns.histplot(x=grad_cbf_corr, ax=ax[2])
ax[2].set_title('corr(grad,cbf)')
sns.histplot(x=grad_reho_corr, ax=ax[3])
ax[3].set_title('corr(grad,reho)')
sns.histplot(x=grad_alff_corr, ax=ax[4])
ax[4].set_title('corr(grad,alff)')
f.savefig(os.path.join(environment.figdir, 'subjects_brainmap_corr'), dpi=150, bbox_inches='tight',
          pad_inches=0.1)
plt.close()

# %% correlations between gradient and subject maps
n_subs = environment.df.shape[0]
ct_reho_corr = np.zeros(n_subs)
cbf_reho_corr = np.zeros(n_subs)

for i in tqdm(np.arange(environment.df.shape[0])):
    ct_reho_corr[i] = sp.stats.spearmanr(load_ct.ct[i, :], load_reho.reho[i, :])[0]
    cbf_reho_corr[i] = sp.stats.spearmanr(load_cbf.cbf[i, :], load_reho.reho[i, :])[0]

f, ax = plt.subplots(1, 2, figsize=(10, 4))
sns.histplot(x=ct_reho_corr, ax=ax[0])
ax[0].set_title('corr(ct,reho)')
sns.histplot(x=cbf_reho_corr, ax=ax[1])
ax[1].set_title('corr(cbf,reho)')
f.savefig(os.path.join(environment.figdir, 'subjects_brainmap_corr_2'), dpi=150, bbox_inches='tight',
          pad_inches=0.1)
plt.close()