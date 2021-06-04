# %%
import sys, os, platform
if platform.system() == 'Linux':
    sys.path.extend(['/cbica/home/parkesl/research_projects/pfactor_gradients'])
from pfactor_gradients.pnc import Environment, Subject
from pfactor_gradients.routines import LoadSC, LoadCT, LoadRLFP, LoadCBF, LoadREHO, LoadALFF, LoadAverageBrainMaps
from pfactor_gradients.pipelines import ComputeGradients, ComputeMinimumControlEnergy
from pfactor_gradients.imaging_derivs import DataVector
from pfactor_gradients.hcp import BrainMapLoader

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
    'cbf': LoadCBF(environment=environment, Subject=Subject)
}

load_average_bms = LoadAverageBrainMaps(loaders_dict=loaders_dict)
load_average_bms.run(return_descending=False)

# append hcp myelin map
hcp_brain_maps = BrainMapLoader(computer=computer)
hcp_brain_maps.load_myelin(lh_annot_file=environment.lh_annot_file, rh_annot_file=environment.rh_annot_file)

data = DataVector(data=hcp_brain_maps.myelin, name='myelin')
data.rankdata()
data.rescale_unit_interval()
load_average_bms.brain_maps['myelin'] = data

# rename brain maps to append '_mean'
for key in load_average_bms.brain_maps:
    load_average_bms.brain_maps[key].name = '{0}-mean'.format(key)

# %% compute minimum energy

# get subject A matrix out
A = load_sc.A[:, :, sge_task_id].copy()
file_prefix = '{0}_'.format(load_sc.df.index[sge_task_id])
print(load_sc.df.index[sge_task_id])
n_subsamples = 0

nct_pipeline = ComputeMinimumControlEnergy(environment=environment, A=A,
                                           states=compute_gradients.grad_bins, n_subsamples=n_subsamples,
                                           control='minimum_fast', T=1, B='wb', file_prefix=file_prefix,
                                           force_rerun=False, save_outputs=True, verbose=True)
nct_pipeline.run()

for key in load_average_bms.brain_maps:
    nct_pipeline = ComputeMinimumControlEnergy(environment=environment, A=A,
                                               states=compute_gradients.grad_bins, n_subsamples=n_subsamples,
                                               control='minimum_fast', T=1, B=load_average_bms.brain_maps[key], file_prefix=file_prefix,
                                               force_rerun=False, save_outputs=True, verbose=True)
    nct_pipeline.run()
