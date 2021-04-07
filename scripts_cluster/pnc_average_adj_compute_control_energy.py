# %%
import sys
sys.path.extend(['/cbica/home/parkesl/research_projects/pfactor_gradients'])
from data_loader.pnc import Environment, Subject
from data_loader.routines import LoadSC, LoadAverageSC, LoadCT, LoadRLFP
from data_loader.pipelines import ComputeGradients, ComputeMinimumControlEnergy

# %% Setup project environment
computer = 'cbica'
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

# %% get mean structural A matrix
filters = {'healthExcludev2': 0, 't1Exclude': 0,
           'b0ProtocolValidationStatus': 1, 'dti64ProtocolValidationStatus': 1, 'dti64Exclude': 0,
           'psychoactiveMedPsychv2': 0}
environment.load_metadata(filters)

# Load sc data
load_sc = LoadSC(environment=environment, Subject=Subject)
load_sc.run()

spars_thresh = 0.06
load_average_sc = LoadAverageSC(load_sc=load_sc, spars_thresh=spars_thresh)
load_average_sc.run()

# %% get control energy
file_prefix = 'average_adj_n-{0}_s-{1}_'.format(load_average_sc.load_sc.df.shape[0], spars_thresh)

# %% whole brain control
nct_pipeline = ComputeMinimumControlEnergy(environment=environment, A=load_average_sc.A,
                                           states=compute_gradients.kmeans.labels_,
                                           control='minimum', T=1, B='wb', file_prefix=file_prefix)
nct_pipeline.run()
