# %%
import sys, os, platform
from oct2py import octave
if platform.system() == 'Linux':
    sys.path.extend(['/cbica/home/parkesl/research_projects/pfactor_gradients'])
    sys.path.append('/usr/bin/octave') # octave install path
    octave.addpath('/gpfs/fs001/cbica/home/parkesl/research_projects/pfactor_gradients/geomsurr') # path to matlab functions
elif platform.system() == 'Darwin':
    sys.path.append('usr/local/bin/octave') # octave install path
    octave.addpath('/Users/lindenmp/Google-Drive-Penn/work/research_projects/pfactor_gradients/geomsurr') # path to matlab functions

from pfactor_gradients.pnc import Environment, Subject
from pfactor_gradients.routines import LoadSC
from pfactor_gradients.pipelines import ComputeGradients, DCM
from pfactor_gradients.utils import get_states_from_gradient

from tqdm import tqdm
import numpy as np
import scipy as sp
import pandas as pd

# %% Setup project environment
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='white', context='paper', font_scale=1)
if platform.system() == 'Linux':
    computer = 'cbica'
elif platform.system() == 'Darwin':
    computer = 'macbook'

    import matplotlib.font_manager as font_manager
    fontpath = '/Users/lindenmp/Library/Fonts/PublicSans-Thin.ttf'
    prop = font_manager.FontProperties(fname=fontpath)
    plt.rcParams['font.family'] = prop.get_name()
    plt.rcParams['svg.fonttype'] = 'none'

parc = 'schaefer'
n_parcels = 400
sc_edge_weight = 'streamlineCount'
environment = Environment(computer=computer, parc=parc, n_parcels=n_parcels, sc_edge_weight=sc_edge_weight)
environment.make_output_dirs()
environment.load_parc_data()

# filter subjects
filters = {'healthExcludev2': 0, 'psychoactiveMedPsychv2': 0,
           't1Exclude': 0, 'fsFinalExclude': 0,
           'b0ProtocolValidationStatus': 1, 'dti64ProtocolValidationStatus': 1, 'dti64Exclude': 0,
           'restProtocolValidationStatus': 1, 'restExclude': 0} # need to add these filters in if doing funcg1 below
environment.load_metadata(filters)

# %% get states
which_grad = 'histg2'

if which_grad == 'histg2':
    if computer == 'macbook':
        gradient = np.loadtxt('/Volumes/T7/research_data/BigBrainWarp/spaces/fsaverage/Hist_G2_Schaefer2018_400Parcels_17Networks.txt')
    elif computer == 'cbica':
        gradient = np.loadtxt('/cbica/home/parkesl/research_data/BigBrainWarp/spaces/fsaverage/Hist_G2_Schaefer2018_400Parcels_17Networks.txt')
    gradient = gradient * -1
elif which_grad == 'funcg1':
    # compute function gradient
    compute_gradients = ComputeGradients(environment=environment, Subject=Subject)
    compute_gradients.run()
    gradient = compute_gradients.gradients[:, 0]

n_bins = int(n_parcels/10)
states = get_states_from_gradient(gradient=gradient, n_bins=n_bins)
n_states = len(np.unique(states))

mask = ~np.eye(n_states, dtype=bool)
indices = np.where(mask)
indices_upper = np.triu_indices(n_states, k=1)
indices_lower = np.tril_indices(n_states, k=-1)

# %% Load sc data
load_sc = LoadSC(environment=environment, Subject=Subject)
load_sc.run()
# refilter environment due to LoadSC excluding on disconnected nodes
environment.df = load_sc.df.copy()
# environment.df = environment.df.iloc[:100, :]
n_subs = environment.df.shape[0]

# %% DCM
dcm = DCM(environment=environment, Subject=Subject, states=states, force_rerun=True)
# dcm.run_mean_ts()
# dcm.run_concat_ts()
dcm.run_concat_mean_ts()

# %%
f, ax = plt.subplots(1, 1, figsize=(15, 4))
sns.heatmap(dcm.rsts_states.transpose(), cmap='gray', ax=ax, center=0)
ax.tick_params(pad=-2.5)
f.savefig(os.path.join(environment.figdir, 'spdcm_ts.png'), dpi=500, bbox_inches='tight',
          pad_inches=0.1)
plt.close()

# %%
# rsts = dcm.rsts_states
# print(rsts.shape)

# spmdir = '/Users/lindenmp/Google-Drive-Penn/work/matlab_tools/spm12'
# outdir = '/Users/lindenmp/Google-Drive-Penn/work/research_projects/pfactor_gradients/output_cluster/pnc/schaefer_400_streamlineCount/pipelines/spdcm'

# octave.eval("rand('state',%i)" % 1)
# octave.spdcm_firstlevel_loop(spmdir, rsts, environment.rsfmri_tr, environment.rsfmri_te, outdir)
