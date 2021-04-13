# %%
import sys, os, platform
if platform.system() == 'Linux':
    sys.path.extend(['/cbica/home/parkesl/research_projects/pfactor_gradients'])
from pfactor_gradients.pnc import Environment, Subject
from pfactor_gradients.routines import LoadSC, LoadAverageSC, LoadCT, LoadRLFP, LoadCBF
from pfactor_gradients.pipelines import ComputeGradients, ComputeMinimumControlEnergy
from pfactor_gradients.imaging_derivs import DataVector
import numpy as np

# %% Setup project environment
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='white', context='talk', font_scale=1)
if platform.system() == 'Linux':
    computer = 'cbica'
    sge_task_id = int(os.getenv("SGE_TASK_ID"))
elif platform.system() == 'Darwin':
    computer = 'macbook'
    sge_task_id = 1

    import matplotlib.font_manager as font_manager
    fontpath = '/Users/lindenmp/Library/Fonts/PublicSans-Thin.ttf'
    prop = font_manager.FontProperties(fname=fontpath)
    plt.rcParams['font.family'] = prop.get_name()
    plt.rcParams['svg.fonttype'] = 'none'
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

spars_thresh = 0.06
load_average_sc = LoadAverageSC(load_sc=load_sc, spars_thresh=spars_thresh)
load_average_sc.run()

# %% get control energy
file_prefix = 'average_adj_n-{0}_s-{1}_'.format(load_average_sc.load_sc.df.shape[0], spars_thresh)
n_subsamples = 20

# flip brain maps
descending = True

# %%
if sge_task_id == 1:
    nct_pipeline = ComputeMinimumControlEnergy(environment=environment, A=load_average_sc.A,
                                               states=compute_gradients.kmeans.labels_, n_subsamples=n_subsamples,
                                               control='minimum_fast', T=1, B='wb', file_prefix=file_prefix)
    nct_pipeline.run()
# %%
elif sge_task_id == 2:
    # load ct data
    load_ct = LoadCT(environment=environment, Subject=Subject)
    load_ct.run()

    if descending:
        ct = DataVector(data=np.nanmean(load_ct.ct, axis=0), name='ct-d')
    else:
        ct = DataVector(data=np.nanmean(load_ct.ct, axis=0), name='ct')
    ct.rankdata(descending=descending)
    ct.rescale_unit_interval()
    ct.brain_surface_plot(environment)

    nct_pipeline = ComputeMinimumControlEnergy(environment=environment, A=load_average_sc.A,
                                               states=compute_gradients.kmeans.labels_, n_subsamples=n_subsamples,
                                               control='minimum_fast', T=1, B=ct, file_prefix=file_prefix)
    nct_pipeline.run()
# %%
elif sge_task_id == 3:
    # load rlfp data
    load_rlfp = LoadRLFP(environment=environment, Subject=Subject)
    load_rlfp.run()

    if descending:
        rlfp = DataVector(data=np.nanmean(load_rlfp.rlfp, axis=0), name='rlfp-d')
    else:
        rlfp = DataVector(data=np.nanmean(load_rlfp.rlfp, axis=0), name='rlfp')
    rlfp.rankdata(descending=descending)
    rlfp.rescale_unit_interval()
    rlfp.brain_surface_plot(environment)

    nct_pipeline = ComputeMinimumControlEnergy(environment=environment, A=load_average_sc.A,
                                               states=compute_gradients.kmeans.labels_, n_subsamples=n_subsamples,
                                               control='minimum_fast', T=1, B=rlfp, file_prefix=file_prefix)
    nct_pipeline.run()
# %%
elif sge_task_id == 4:
    # load cbf data
    load_cbf = LoadCBF(environment=environment, Subject=Subject)
    load_cbf.run()

    if descending:
        cbf = DataVector(data=np.nanmean(load_cbf.cbf, axis=0), name='cbf-d')
    else:
        cbf = DataVector(data=np.nanmean(load_cbf.cbf, axis=0), name='cbf')
    cbf.rankdata(descending=descending)
    cbf.rescale_unit_interval()
    cbf.brain_surface_plot(environment)

    nct_pipeline = ComputeMinimumControlEnergy(environment=environment, A=load_average_sc.A,
                                               states=compute_gradients.kmeans.labels_, n_subsamples=n_subsamples,
                                               control='minimum_fast', T=1, B=cbf, file_prefix=file_prefix)
    nct_pipeline.run()
# %%
elif sge_task_id == 5:
    nct_pipeline = ComputeMinimumControlEnergy(environment=environment, A=load_average_sc.A,
                                               states=compute_gradients.kmeans.labels_, n_subsamples=n_subsamples,
                                               control='minimum_fast', T=1, B='wb', file_prefix=file_prefix,
                                               add_noise=True)
    nct_pipeline.run()
