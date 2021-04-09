# %%
import sys, os, platform
if platform.system() == 'Linux':
    sys.path.extend(['/cbica/home/parkesl/research_projects/pfactor_gradients'])
from pfactor_gradients.pnc import Environment, Subject
from pfactor_gradients.routines import LoadSC, LoadAverageSC, LoadCT, LoadRLFP
from pfactor_gradients.pipelines import ComputeGradients, ComputeMinimumControlEnergy
from pfactor_gradients.imaging_derivs import DataVector
import numpy as np

# %% Setup project environment
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='white', context='talk', font_scale=1)
if platform.system() == 'Linux':
    computer = 'cbica'
    sge_task_id = int(os.getenv("SGE_TASK_ID"))-1
elif platform.system() == 'Darwin':
    computer = 'macbook'
    sge_task_id = 0

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

# load ct data
load_ct = LoadCT(environment=environment, Subject=Subject)
load_ct.run()

ct = DataVector(data=np.nanmean(load_ct.ct, axis=0), name='ct')
ct.rankdata()
ct.rescale_unit_interval()
ct.shuffle_data(shuffle_indices=environment.spun_indices)

ct_null = DataVector(data=ct.data_shuf[:, sge_task_id], name='ct-null-{0}'.format(sge_task_id))

nct_pipeline = ComputeMinimumControlEnergy(environment=environment, A=load_average_sc.A,
                                           states=compute_gradients.kmeans.labels_, n_subsamples=n_subsamples,
                                           control='minimum_fast', T=1, B=ct_null, file_prefix=file_prefix,
                                           force_rerun=False, save_outputs=True, verbose=True)
nct_pipeline.run()
