# %% import
import sys, os, platform
if platform.system() == 'Linux':
    sys.path.extend(['/cbica/home/parkesl/research_projects/nct_hierarchy'])
from src.pnc import Environment, Subject
from src.routines import LoadSC, LoadAverageSC
from src.pipelines import ComputeGradients
from src.utils import get_states_from_brain_map
from src.imaging_derivs import DataVector
from src.brain_maps import BrainMapLoader

import numpy as np

workspace = os.getenv("MY_PYTHON_WORKSPACE")
print('Running workspace: {0}'.format(workspace))
which_brain_map = os.getenv("WHICH_BRAIN_MAP")
print('Brain map: {0}'.format(which_brain_map))
intrahemi = os.getenv("INTRAHEMI") == 'True'
print('Intra-hemisphere: {0}'.format(intrahemi))

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

# %% compute functional gradient
compute_gradients = ComputeGradients(environment=environment, Subject=Subject)
compute_gradients.run()

# %%
brain_map_loader = BrainMapLoader(computer=computer, n_parcels=n_parcels)

# %% run workspace specific lines
if workspace == 'ave_adj':
    # get average sc data
    consist_thresh = 0.6

    load_average_sc = LoadAverageSC(load_sc=load_sc, consist_thresh=consist_thresh)
    load_average_sc.run()
    A = load_average_sc.A.copy()
    if intrahemi == True:
        A = A[:int(n_parcels / 2), :int(n_parcels / 2)]

    # brain_maps = load_average_bms.brain_maps.copy()
    brain_maps = dict()

    # append fc gradient to brain maps
    dv = DataVector(data=compute_gradients.gradients[:, 0], name='func-g1')
    if intrahemi == True:
        dv.data = dv.data[:int(n_parcels / 2)]
    dv.rankdata()
    dv.rescale_unit_interval()
    brain_maps[dv.name] = dv

    # append tau map from Gao et al. 2020 eLife
    if parc == 'schaefer':
        brain_map_loader.load_tau(return_log=False)
        dv = DataVector(data=brain_map_loader.tau, name='tau')
        if intrahemi == True:
            dv.data = dv.data[:int(n_parcels / 2)]
        dv.rankdata()
        dv.rescale_unit_interval()
        brain_maps[dv.name] = dv
elif workspace == 'subj_adj':
    pass

# %% get states
if which_brain_map == 'hist-g2':
    brain_map_loader.load_cyto()
    state_brain_map = brain_map_loader.cyto
    state_brain_map = state_brain_map * -1
elif which_brain_map == 'func-g1':
    state_brain_map = compute_gradients.gradients[:, 0].copy()

# bin_size = 9
bin_size = 10
# bin_size = 11
if intrahemi == True:
    state_brain_map = state_brain_map[:int(n_parcels / 2)]
    n_bins = int(int(n_parcels / 2) / bin_size)
else:
    n_bins = int(n_parcels / bin_size)

states = get_states_from_brain_map(brain_map=state_brain_map, n_bins=n_bins)
n_states = len(np.unique(states))
mask = ~np.eye(n_states, dtype=bool)
indices = np.where(mask)
indices_upper = np.triu_indices(n_states, k=1)
indices_lower = np.tril_indices(n_states, k=-1)

print('dti64QAManualScore', np.sum(environment.df['dti64QAManualScore'] == 2))
print('averageManualRating', np.sum(environment.df['averageManualRating'] == 2))
