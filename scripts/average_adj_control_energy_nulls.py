# %%
import sys, os, platform
from oct2py import octave
if platform.system() == 'Linux':
    sys.path.extend(['/cbica/home/parkesl/research_projects/pfactor_gradients'])
    sys.path.append('/usr/bin/octave') # octave install path
    octave.addpath('/gpfs/fs001/cbica/home/parkesl/research_projects/pfactor_gradients/geomsurr') # path to matlab functions
elif platform.system() == 'Darwin':
    sys.path.append('usr/local/bin/octave') # octave install path
    octave.addpath('/Users/lindenmp/Google-Drive-Penn/work/research_projects/pfactor_gradients/geomsurr') # path to matlab functions

from pfactor_gradients.pnc import Environment, Subject
from pfactor_gradients.routines import LoadSC, LoadCT, LoadRLFP, LoadCBF, LoadREHO, LoadALFF,\
    LoadAverageSC, LoadAverageBrainMaps
from pfactor_gradients.pipelines import ComputeGradients, ComputeMinimumControlEnergy
from pfactor_gradients.imaging_derivs import DataVector
from pfactor_gradients.hcp import BrainMapLoader
import scipy as sp
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
n_bins = int(n_parcels/10)
compute_gradients = ComputeGradients(environment=environment, Subject=Subject, n_bins=n_bins)
compute_gradients.run()

# %% Load sc data
load_sc = LoadSC(environment=environment, Subject=Subject)
load_sc.run()
# refilter environment due to LoadSC excluding on disconnected nodes
environment.df = load_sc.df.copy()

if parc == 'schaefer' and n_parcels == 400:
    spars_thresh = 0.06
elif parc == 'schaefer' and n_parcels == 200:
    spars_thresh = 0.12
elif parc == 'glasser' and n_parcels == 360:
    spars_thresh = 0.07
load_average_sc = LoadAverageSC(load_sc=load_sc, spars_thresh=spars_thresh)
load_average_sc.run()
A = load_average_sc.A.copy()

# rewire mean adjacency matrix
D = sp.spatial.distance.pdist(environment.centroids, 'euclidean')
D = sp.spatial.distance.squareform(D)
octave.eval("rand('state',%i)" % sge_task_id)
Wwp, Wsp, Wssp = octave.geomsurr(A, D, 3, 2, nout=3)

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

load_average_bms.brain_maps.pop('ct')
load_average_bms.brain_maps.pop('cbf')

# %% get control energy
file_prefix = 'average_adj_n-{0}_s-{1}_'.format(load_average_sc.load_sc.df.shape[0], spars_thresh)
n_subsamples = 0

# %% brain map null (spin test)
for key in load_average_bms.brain_maps:
    load_average_bms.brain_maps[key].shuffle_data(shuffle_indices=environment.spun_indices)

    permuted_bm = DataVector(data=load_average_bms.brain_maps[key].data_shuf[:, sge_task_id].copy(),
                             name='{0}-spin-{1}'.format(key, sge_task_id))

    nct_pipeline = ComputeMinimumControlEnergy(environment=environment, A=A,
                                               states=compute_gradients.grad_bins, n_subsamples=n_subsamples,
                                               control='minimum_fast', T=1, B=permuted_bm, file_prefix=file_prefix,
                                               force_rerun=True, save_outputs=True, verbose=True)
    nct_pipeline.run()

# %% brain map null (random)
# for key in load_average_bms.brain_maps:
#     load_average_bms.brain_maps[key].shuffle_data(n_shuffles=10000)
#
#     permuted_bm = DataVector(data=load_average_bms.brain_maps[key].data_shuf[:, sge_task_id].copy(),
#                              name='{0}-rand-{1}'.format(key, sge_task_id))
#
#     nct_pipeline = ComputeMinimumControlEnergy(environment=environment, A=A,
#                                                states=compute_gradients.grad_bins, n_subsamples=n_subsamples,
#                                                control='minimum_fast', T=1, B=permuted_bm, file_prefix=file_prefix,
#                                                force_rerun=True, save_outputs=True, verbose=True)
#     nct_pipeline.run()

# %% random b map
# np.random.seed(sge_task_id)
# permuted_bm = DataVector(data=np.random.uniform(low=0, high=1, size=environment.n_parcels),
#                          name='runi-{0}'.format(sge_task_id))
#
# nct_pipeline = ComputeMinimumControlEnergy(environment=environment, A=A,
#                                            states=compute_gradients.grad_bins, n_subsamples=n_subsamples,
#                                            control='minimum_fast', T=1, B=permuted_bm, file_prefix=file_prefix,
#                                            force_rerun=True, save_outputs=True, verbose=True)
# nct_pipeline.run()

# %% network null
A_list = [Wwp, Wsp, Wssp]
file_prefixes = ['average_adj_n-{0}_s-{1}_null-mni-wwp-{2}_'.format(load_average_sc.load_sc.df.shape[0], spars_thresh, sge_task_id),
                 'average_adj_n-{0}_s-{1}_null-mni-wsp-{2}_'.format(load_average_sc.load_sc.df.shape[0], spars_thresh, sge_task_id),
                 'average_adj_n-{0}_s-{1}_null-mni-wssp-{2}_'.format(load_average_sc.load_sc.df.shape[0], spars_thresh, sge_task_id)]

for A_idx, A_entry in enumerate(A_list):
    file_prefix = file_prefixes[A_idx]

    nct_pipeline = ComputeMinimumControlEnergy(environment=environment, A=A_entry,
                                               states=compute_gradients.grad_bins, n_subsamples=n_subsamples,
                                               control='minimum_fast', T=1, B='wb', file_prefix=file_prefix,
                                               force_rerun=True, save_outputs=True, verbose=True)
    nct_pipeline.run()

    for key in load_average_bms.brain_maps:
        nct_pipeline = ComputeMinimumControlEnergy(environment=environment, A=A_entry,
                                                   states=compute_gradients.grad_bins, n_subsamples=n_subsamples,
                                                   control='minimum_fast', T=1, B=load_average_bms.brain_maps[key], file_prefix=file_prefix,
                                                   force_rerun=True, save_outputs=True, verbose=True)
        nct_pipeline.run()
