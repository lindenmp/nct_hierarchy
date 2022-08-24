# %% import
import sys, os, platform
if platform.system() == 'Linux':
    sys.path.extend(['/cbica/home/parkesl/research_projects/nct_hierarchy'])
from src.pnc import Environment, Subject
from src.routines import LoadSC, LoadAverageSC
from src.brain_maps import BrainMapLoader

import numpy as np

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

# %% Load average A matrix

# first, get subject A matrices
# note, this performs more subject exclusion

load_sc = LoadSC(environment=environment, Subject=Subject)
load_sc.run()
# refilter environment due to LoadSC excluding on disconnected nodes
environment.df = load_sc.df.copy()
n_subs = environment.df.shape[0]

# second, get average sc data using consistency thresholding
consist_thresh = 0.6

load_average_sc = LoadAverageSC(load_sc=load_sc, consist_thresh=consist_thresh)
load_average_sc.run()
A = load_average_sc.A.copy()

# %% get states
brain_map_loader = BrainMapLoader(computer=computer, n_parcels=n_parcels)
brain_map_loader.load_cyto()
state_brain_map = brain_map_loader.cyto
state_brain_map = state_brain_map * -1

# %% save data
np.save('/Users/lindenmp/Google-Drive-Penn/work/research_projects/nct_hierarchy/data/A', A)
np.save('/Users/lindenmp/Google-Drive-Penn/work/research_projects/nct_hierarchy/data/sf_axis', state_brain_map)
environment.centroids.to_csv('/Users/lindenmp/Google-Drive-Penn/work/research_projects/nct_hierarchy/data/parcellation_centroids.csv')
