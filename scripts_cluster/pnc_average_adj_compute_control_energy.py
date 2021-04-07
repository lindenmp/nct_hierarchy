# %%
import sys, os
sys.path.extend(['/cbica/home/parkesl/research_projects/pfactor_gradients'])
from data_loader.pnc import Environment, Subject
from data_loader.routines import LoadSC, LoadAverageSC, LoadCT, LoadRLFP
from data_loader.pipelines import ComputeGradients, ComputeMinimumControlEnergy
from utils.imaging_derivs import DataVector
import numpy as np

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

# %% Load sc data
load_sc = LoadSC(environment=environment, Subject=Subject)
load_sc.run()

spars_thresh = 0.06
load_average_sc = LoadAverageSC(load_sc=load_sc, spars_thresh=spars_thresh)
load_average_sc.run()

# %% get control energy
file_prefix = 'average_adj_n-{0}_s-{1}_'.format(load_average_sc.load_sc.df.shape[0], spars_thresh)
sge_task_id = int(os.getenv("SGE_TASK_ID"))

if sge_task_id == 1:
    nct_pipeline = ComputeMinimumControlEnergy(environment=environment, A=load_average_sc.A,
                                               states=compute_gradients.kmeans.labels_,
                                               control='minimum', T=1, B='wb', file_prefix=file_prefix)
elif sge_task_id == 2:
    # load ct data
    load_ct = LoadCT(environment=environment, Subject=Subject)
    load_ct.run()

    ct = DataVector(data=np.nanmean(load_ct.ct, axis=0), name='ct')
    ct.rankdata()
    ct.rescale_unit_interval()
    ct.brain_surface_plot(environment)

    nct_pipeline = ComputeMinimumControlEnergy(environment=environment, A=load_average_sc.A,
                                               states=compute_gradients.kmeans.labels_,
                                               control='minimum', T=1, B=ct, file_prefix=file_prefix)
elif sge_task_id == 3:
    # load rlfp data
    load_rlfp = LoadRLFP(environment=environment, Subject=Subject)
    load_rlfp.run()

    rlfp = DataVector(data=np.nanmean(load_rlfp.rlfp, axis=0), name='rlfp')
    rlfp.rankdata()
    rlfp.rescale_unit_interval()
    rlfp.brain_surface_plot(environment)

    nct_pipeline = ComputeMinimumControlEnergy(environment=environment, A=load_average_sc.A,
                                               states=compute_gradients.kmeans.labels_,
                                               control='minimum', T=1, B=rlfp, file_prefix=file_prefix)

nct_pipeline.run()
