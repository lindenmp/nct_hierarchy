# %% import
import sys, os, platform
if platform.system() == 'Linux':
    sys.path.extend(['/cbica/home/parkesl/research_projects/pfactor_gradients'])
from pfactor_gradients.pnc import Environment, Subject
from pfactor_gradients.routines import LoadSC, LoadCT, LoadSA, LoadAverageSC, LoadAverageBrainMaps, LoadRLFP, LoadALFF
from pfactor_gradients.pipelines import ComputeGradients
from pfactor_gradients.utils import get_states_from_brain_map
from pfactor_gradients.imaging_derivs import DataVector

import numpy as np
import pandas as pd

# %% Setup project environment
if platform.system() == 'Linux':
    computer = 'cbica'
elif platform.system() == 'Darwin':
    computer = 'macbook'
parc = 'schaefer'
n_parcels = 200
sc_edge_weight = 'streamlineCount'
environment = Environment(computer=computer, parc=parc, n_parcels=n_parcels, sc_edge_weight=sc_edge_weight)
environment.make_output_dirs()
environment.load_parc_data()

# filter subjects
filters = {'healthExcludev2': 0, 'psychoactiveMedPsychv2': 0,
           't1Exclude': 0, 'fsFinalExclude': 0,
           'b0ProtocolValidationStatus': 1, 'dti64ProtocolValidationStatus': 1, 'dti64Exclude': 0,
           'restProtocolValidationStatus': 1, 'restExclude': 0} # need to add these filters in if doing func-g1 below
environment.load_metadata(filters)

# %% Load sc data
# note, this performs more subject exclusion
load_sc = LoadSC(environment=environment, Subject=Subject)
load_sc.run()
# refilter environment due to LoadSC excluding on disconnected nodes
environment.df = load_sc.df.copy()
n_subs = environment.df.shape[0]

consist_thresh = 0.6

load_average_sc = LoadAverageSC(load_sc=load_sc, consist_thresh=consist_thresh)
load_average_sc.run()
A = load_average_sc.A.copy()

# %% load mean brain maps
loaders_dict = {
    # 'ct': LoadCT(environment=environment, Subject=Subject),
    # 'sa': LoadSA(environment=environment, Subject=Subject),
    'rlfp': LoadRLFP(environment=environment, Subject=Subject),
    'alff': LoadALFF(environment=environment, Subject=Subject)
}

load_average_bms = LoadAverageBrainMaps(loaders_dict=loaders_dict)
load_average_bms.run(return_descending=False)

# %% compute functional gradient
compute_gradients = ComputeGradients(environment=environment, Subject=Subject)
compute_gradients.run()

# append fc gradient to brain maps
dv = DataVector(data=compute_gradients.gradients[:, 0], name='func-g1')
dv.rankdata()
dv.rescale_unit_interval()
load_average_bms.brain_maps[dv.name] = dv

# %% append tau map from Gao et al. 2020 eLife
df_human_tau = pd.read_csv('/Users/lindenmp/research_data/field-echos/tau_Schaefer2018_{0}Parcels_17Networks.csv'
                           .format(n_parcels), index_col=0)
nan_mask = df_human_tau['tau'].isna()

dv = DataVector(data=df_human_tau['tau'].values, name='tau')
dv.rankdata()
dv.rescale_unit_interval()
load_average_bms.brain_maps[dv.name] = dv

# %% append hcp myelin map
# hcp_brain_maps = BrainMapLoader(computer=computer)
# hcp_brain_maps.load_myelin(lh_annot_file=environment.lh_annot_file, rh_annot_file=environment.rh_annot_file)
#
# data = DataVector(data=hcp_brain_maps.myelin, name='myelin')
# data.rankdata()
# data.rescale_unit_interval()
# load_average_bms.brain_maps['myelin'] = data

# %% get states
which_brain_map = 'hist-g2'
# which_brain_map = 'hist-g1'
# which_brain_map = 'func-g1'
# which_brain_map = 'ct'

if computer == 'macbook':
    # bbw_dir = '/Volumes/T7/research_data/BigBrainWarp/spaces/fsaverage/'
    bbw_dir = '/Users/lindenmp/research_data/BigBrainWarp/spaces/fsaverage/'
elif computer == 'cbica':
    bbw_dir = '/cbica/home/parkesl/research_data/BigBrainWarp/spaces/fsaverage/'

if which_brain_map == 'hist-g1':
    if parc == 'schaefer':
        state_brain_map = np.loadtxt(os.path.join(bbw_dir, 'Hist_G1_Schaefer2018_{0}Parcels_17Networks.txt' \
                                                  .format(n_parcels)))
    elif parc == 'glasser':
        state_brain_map = np.loadtxt(os.path.join(bbw_dir, 'Hist_G1_HCP-MMP1.txt'))
    state_brain_map = state_brain_map * -1
elif which_brain_map == 'hist-g2':
    if parc == 'schaefer':
        state_brain_map = np.loadtxt(os.path.join(bbw_dir, 'Hist_G2_Schaefer2018_{0}Parcels_17Networks.txt' \
                                                  .format(n_parcels)))
    elif parc == 'glasser':
        state_brain_map = np.loadtxt(os.path.join(bbw_dir, 'Hist_G2_HCP-MMP1.txt'))
    state_brain_map = state_brain_map * -1
elif which_brain_map == 'func-g1':
    state_brain_map = compute_gradients.gradients[:, 0].copy()
else:
    state_brain_map = load_average_bms.brain_maps[which_brain_map].data.copy()

n_bins = int(n_parcels / 10)
states = get_states_from_brain_map(brain_map=state_brain_map, n_bins=n_bins)
n_states = len(np.unique(states))
mask = ~np.eye(n_states, dtype=bool)
indices = np.where(mask)
indices_upper = np.triu_indices(n_states, k=1)
indices_lower = np.tril_indices(n_states, k=-1)

print('dti64QAManualScore', np.sum(environment.df['dti64QAManualScore'] == 2))
print('averageManualRating', np.sum(environment.df['averageManualRating'] == 2))
