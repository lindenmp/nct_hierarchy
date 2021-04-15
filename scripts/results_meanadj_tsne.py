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
indices = np.triu_indices(n_states, k=1)

# %% get minimum control energy
my_list = ['wb',] + list(load_average_bms.brain_maps.keys())
E = np.zeros((n_states, n_states, len(my_list)))
df = pd.DataFrame()

for i, val in enumerate(my_list):
    file = 'average_adj_n-775_s-0.06_ns-40-0_c-minimum_fast_T-1_B-{0}_E.npy'.format(val)
    E[:, :, i] = np.load(os.path.join(environment.pipelinedir, 'minimum_control_energy', file))
    df[val] = E[:, :, i][indices]

# %% MDS
from sklearn.manifold import MDS, TSNE, Isomap, SpectralEmbedding

np.random.seed(0)
X = df.drop(labels='wb', axis=1)
X = (X - X.mean()) / X.std()
embedding = TSNE(n_components=2)
X_transformed = embedding.fit_transform(X)

# %%
f, ax = plt.subplots(1, 1, figsize=(5, 5))
state_idx = np.tile(np.arange(n_states), (n_states,1))
state_idx = state_idx[indices]
ax.scatter(x=X_transformed[:, 0], y=X_transformed[:, 1], c=state_idx, cmap='tab20', s=5, alpha=0.5)
f.savefig(os.path.join(environment.figdir, 'tsne_states.png'), dpi=150, bbox_inches='tight',
          pad_inches=0.1)
plt.close()

# %%
n_maps = len(load_average_bms.brain_maps)
f, ax = plt.subplots(1, n_maps, figsize=(n_maps*5, 5))
for i, key in enumerate(load_average_bms.brain_maps):
    state_idx = load_average_bms.brain_maps[key].data_mean
    state_idx = state_idx[indices]
    ax[i].scatter(x=X_transformed[:, 0], y=X_transformed[:, 1], c=state_idx, cmap='hot', s=5, alpha=0.5)
    f.savefig(os.path.join(environment.figdir, 'tnse_brain_maps.png'), dpi=150, bbox_inches='tight',
              pad_inches=0.1)
    plt.close()
