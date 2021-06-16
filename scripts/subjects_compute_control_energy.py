# %%
import sys, os, platform
if platform.system() == 'Linux':
    sys.path.extend(['/cbica/home/parkesl/research_projects/pfactor_gradients'])
from pfactor_gradients.pnc import Environment, Subject
from pfactor_gradients.routines import LoadSC, LoadCT, LoadSA
from pfactor_gradients.pipelines import ComputeGradients, ComputeMinimumControlEnergy
from pfactor_gradients.imaging_derivs import DataVector
from pfactor_gradients.utils import get_states_from_gradient
import numpy as np

# %% Setup project environment
if platform.system() == 'Linux':
    computer = 'cbica'
    sge_task_id = int(os.getenv("SGE_TASK_ID")) - 1
elif platform.system() == 'Darwin':
    computer = 'macbook'
    sge_task_id = 0
print(sge_task_id)

parc = 'schaefer'
n_parcels = 400
sc_edge_weight = 'streamlineCount'
environment = Environment(computer=computer, parc=parc, n_parcels=n_parcels, sc_edge_weight=sc_edge_weight)
environment.make_output_dirs()
environment.load_parc_data()

# filter subjects
filters = {'healthExcludev2': 0, 'psychoactiveMedPsychv2': 0,
           't1Exclude': 0, 'fsFinalExclude': 0,
           'b0ProtocolValidationStatus': 1, 'dti64ProtocolValidationStatus': 1, 'dti64Exclude': 0}
           # 'restProtocolValidationStatus': 1, 'restExclude': 0} # need to add these filters in if doing funcg1 below
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

# %% Load sc data
load_sc = LoadSC(environment=environment, Subject=Subject)
load_sc.run()
# get subject A matrix out
A = load_sc.A[:, :, sge_task_id].copy()
print(load_sc.df.index[sge_task_id])

# refilter environment due to LoadSC excluding on disconnected nodes
environment.df = load_sc.df.copy()
# retain ith subject
environment.df = environment.df.iloc[sge_task_id, :].to_frame().transpose()
print(environment.df.index[0])

# %% load mean brain maps
loaders_dict = {
    'ct': LoadCT(environment=environment, Subject=Subject),
    'sa': LoadSA(environment=environment, Subject=Subject)
}

for key in loaders_dict:
    loaders_dict[key].run()

# %% compute minimum energy
file_prefix = '{0}_{1}_'.format(environment.df.index[0], which_grad)
n_subsamples = 0

nct_pipeline = ComputeMinimumControlEnergy(environment=environment, A=A,
                                           states=states, n_subsamples=n_subsamples,
                                           control='minimum_fast', T=1, B='wb', file_prefix=file_prefix,
                                           force_rerun=False, save_outputs=True, verbose=True)
nct_pipeline.run()

for key in loaders_dict:
    try:
        bm = DataVector(data=loaders_dict[key].values[0, :].copy(), name=key)
        bm.rankdata()
        bm.rescale_unit_interval()

        nct_pipeline = ComputeMinimumControlEnergy(environment=environment, A=A,
                                                   states=states, n_subsamples=n_subsamples,
                                                   control='minimum_fast', T=1, B=bm, file_prefix=file_prefix,
                                                   force_rerun=False, save_outputs=True, verbose=True)
        nct_pipeline.run()

        # with flipped brain map
        bm = DataVector(data=loaders_dict[key].values[0, :].copy(), name=key+'_flip')
        bm.rankdata(descending=True)
        bm.rescale_unit_interval()

        nct_pipeline = ComputeMinimumControlEnergy(environment=environment, A=A,
                                                   states=states, n_subsamples=n_subsamples,
                                                   control='minimum_fast', T=1, B=bm, file_prefix=file_prefix,
                                                   force_rerun=False, save_outputs=True, verbose=True)
        nct_pipeline.run()
    except IndexError:
        pass
