# %%
import sys, os, platform
if platform.system() == 'Linux':
    sys.path.extend(['/cbica/home/parkesl/research_projects/pfactor_gradients'])
from pfactor_gradients.pnc import Environment, Subject
from pfactor_gradients.routines import LoadSC, LoadCT, LoadRLFP
from pfactor_gradients.pipelines import ComputeGradients, ComputeMinimumControlEnergy
from pfactor_gradients.imaging_derivs import DataVector

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

# %% load data

# Load sc data
load_sc = LoadSC(environment=environment, Subject=Subject)
load_sc.run()
# get subject A matrix out
A = load_sc.A[:, :, sge_task_id]
print(load_sc.df.index[sge_task_id])

# refilter environment due to LoadSC excluding on disconnected nodes
environment.df = load_sc.df.copy()
# retain ith subject
environment.df = environment.df.iloc[sge_task_id, :].to_frame().transpose()
print(environment.df.index[0])

# Load ct data for ith subject
load_ct = LoadCT(environment=environment, Subject=Subject)
load_ct.run()
ct = DataVector(data=load_ct.ct[0, :], name='ct')
ct.rankdata()
ct.rescale_unit_interval()

# Load ct data for ith subject
load_rlfp = LoadRLFP(environment=environment, Subject=Subject)
load_rlfp.run()
rlfp = DataVector(data=load_rlfp.rlfp[0, :], name='rlfp')
rlfp.rankdata()
rlfp.rescale_unit_interval()

# %% compute minimum energy
file_prefix = '{0}_'.format(environment.df.index[0])
n_subsamples = 20

nct_pipeline = ComputeMinimumControlEnergy(environment=environment, A=A,
                                           states=compute_gradients.kmeans.labels_, n_subsamples=n_subsamples,
                                           control='minimum_fast', T=1, B='wb', file_prefix=file_prefix)
nct_pipeline.run()

nct_pipeline = ComputeMinimumControlEnergy(environment=environment, A=A,
                                           states=compute_gradients.kmeans.labels_, n_subsamples=n_subsamples,
                                           control='minimum_fast', T=1, B=ct, file_prefix=file_prefix)
nct_pipeline.run()

nct_pipeline = ComputeMinimumControlEnergy(environment=environment, A=A,
                                           states=compute_gradients.kmeans.labels_, n_subsamples=n_subsamples,
                                           control='minimum_fast', T=1, B=rlfp, file_prefix=file_prefix)
nct_pipeline.run()
