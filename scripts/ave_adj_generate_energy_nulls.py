# %% import
import sys, os, platform
if platform.system() == 'Linux':
    sge_task_id = int(os.getenv("SGE_TASK_ID"))-1
    sys.path.extend(['/cbica/home/parkesl/research_projects/pfactor_gradients'])
elif platform.system() == 'Darwin':
    sge_task_id = 0
print(sge_task_id)

from pfactor_gradients.pipelines import ComputeMinimumControlEnergy

import scipy as sp
from bct.algorithms.reference import randmio_und

from oct2py import octave
if platform.system() == 'Linux':
    sys.path.append('/usr/bin/octave') # octave install path
    octave.addpath('/gpfs/fs001/cbica/home/parkesl/research_projects/pfactor_gradients/geomsurr') # path to matlab functions
elif platform.system() == 'Darwin':
    sys.path.append('usr/local/bin/octave') # octave install path
    octave.addpath('/Users/lindenmp/Google-Drive-Penn/work/research_projects/pfactor_gradients/geomsurr') # path to matlab functions

# %% import workspace
from setup_workspace_ave_adj import *

# %% rewire mean adjacency matrix with spatial constraints
D = sp.spatial.distance.pdist(environment.centroids, 'euclidean')
D = sp.spatial.distance.squareform(D)
octave.eval("rand('state',%i)" % sge_task_id)
Wwp, Wsp, Wssp = octave.geomsurr(A, D, 3, 2, nout=3)

# rewire mean adjacency matrix without spatial constraints
n_parcels = A.shape[0]
n_connections = n_parcels * n_parcels - n_parcels
n_edge_swaps = int(5 * 10e4)
n_iter = int(n_edge_swaps / n_connections)
np.random.seed(sge_task_id)
R, eff = randmio_und(A, itr=n_iter)

# %% get control energy
file_prefix = 'average_adj_n-{0}_s-{1}_{2}_'.format(load_average_sc.load_sc.df.shape[0], spars_thresh, which_brain_map)
n_subsamples = 0

# %% brain map null (spin test)
# for key in load_average_bms.brain_maps:
#     load_average_bms.brain_maps[key].shuffle_data(shuffle_indices=environment.spun_indices)
#
#     permuted_bm = DataVector(data=load_average_bms.brain_maps[key].data_shuf[:, sge_task_id].copy(),
#                              name='{0}-spin-{1}'.format(key, sge_task_id))
#
#     nct_pipeline = ComputeMinimumControlEnergy(environment=environment, A=A,
#                                                states=states, n_subsamples=n_subsamples,
#                                                control='minimum_fast', T=1, B=permuted_bm, file_prefix=file_prefix,
#                                                force_rerun=True, save_outputs=True, verbose=True)
#     nct_pipeline.run()

# %% brain map null (random)
# for key in load_average_bms.brain_maps:
#     load_average_bms.brain_maps[key].shuffle_data(n_shuffles=10000)
#
#     permuted_bm = DataVector(data=load_average_bms.brain_maps[key].data_shuf[:, sge_task_id].copy(),
#                              name='{0}-rand-{1}'.format(key, sge_task_id))
#
#     nct_pipeline = ComputeMinimumControlEnergy(environment=environment, A=A,
#                                                states=states, n_subsamples=n_subsamples,
#                                                control='minimum_fast', T=1, B=permuted_bm, file_prefix=file_prefix,
#                                                force_rerun=True, save_outputs=True, verbose=True)
#     nct_pipeline.run()

# %% random b map
# np.random.seed(sge_task_id)
# permuted_bm = DataVector(data=np.random.uniform(low=0, high=1, size=environment.n_parcels),
#                          name='runi-{0}'.format(sge_task_id))
#
# nct_pipeline = ComputeMinimumControlEnergy(environment=environment, A=A,
#                                            states=states, n_subsamples=n_subsamples,
#                                            control='minimum_fast', T=1, B=permuted_bm, file_prefix=file_prefix,
#                                            force_rerun=True, save_outputs=True, verbose=True)
# nct_pipeline.run()

# %% network null
A_list = [Wwp, Wsp, Wssp, R] # Wssp
file_prefixes = ['average_adj_n-{0}_s-{1}_{2}_null-mni-wwp-{3}_'.format(load_average_sc.load_sc.df.shape[0],
                                                                        spars_thresh, which_brain_map, sge_task_id),
                 'average_adj_n-{0}_s-{1}_{2}_null-mni-wsp-{3}_'.format(load_average_sc.load_sc.df.shape[0],
                                                                        spars_thresh, which_brain_map, sge_task_id),
                 'average_adj_n-{0}_s-{1}_{2}_null-mni-wssp-{3}_'.format(load_average_sc.load_sc.df.shape[0],
                                                                         spars_thresh, which_brain_map, sge_task_id),
                 'average_adj_n-{0}_s-{1}_{2}_null-nospat-{3}_'.format(load_average_sc.load_sc.df.shape[0],
                                                                       spars_thresh, which_brain_map, sge_task_id)]

for A_idx, A_entry in enumerate(A_list):
    file_prefix = file_prefixes[A_idx]

    nct_pipeline = ComputeMinimumControlEnergy(environment=environment, A=A_entry,
                                               states=states, n_subsamples=n_subsamples,
                                               control='minimum_fast', T=1, B='wb', file_prefix=file_prefix,
                                               force_rerun=True, save_outputs=True, verbose=True)
    nct_pipeline.run()

    # for key in load_average_bms.brain_maps:
    #     nct_pipeline = ComputeMinimumControlEnergy(environment=environment, A=A_entry,
    #                                                states=states, n_subsamples=n_subsamples,
    #                                                control='minimum_fast', T=1, B=load_average_bms.brain_maps[key],
    #                                                file_prefix=file_prefix,
    #                                                force_rerun=True, save_outputs=True, verbose=True)
    #     nct_pipeline.run()
