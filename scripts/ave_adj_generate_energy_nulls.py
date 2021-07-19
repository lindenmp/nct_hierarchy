# %% import
import sys, os, platform
if platform.system() == 'Linux':
    sge_task_id = int(os.getenv("SGE_TASK_ID"))-1
    sys.path.extend(['/cbica/home/parkesl/research_projects/pfactor_gradients'])
elif platform.system() == 'Darwin':
    sge_task_id = 0
print(sge_task_id)

from pfactor_gradients.imaging_derivs import DataMatrix
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
file_prefix = 'average_adj_n-{0}_cthr-{1}_smap-{2}_'.format(load_average_sc.load_sc.df.shape[0],
                                                            consist_thresh, which_brain_map)
# %% network null
A_list = [Wsp, Wssp]
file_prefixes = ['average_adj_n-{0}_cthr-{1}_smap-{2}_null-mni-wsp-{3}_'.format(load_average_sc.load_sc.df.shape[0],
                                                                                consist_thresh, which_brain_map,
                                                                                sge_task_id),
                 'average_adj_n-{0}_cthr-{1}_smap-{2}_null-mni-wssp-{3}_'.format(load_average_sc.load_sc.df.shape[0],
                                                                                 consist_thresh, which_brain_map,
                                                                                 sge_task_id)]

B_dict = dict()
B = DataMatrix(data=np.eye(n_parcels), name='identity')
B_dict[B.name] = B

# for key in load_average_bms.brain_maps:
#     B = DataMatrix(data=np.zeros((n_parcels, n_parcels)), name=key)
#     B.data[np.eye(n_parcels) == 1] = 1 + load_average_bms.brain_maps[key].data
#     B_dict[B.name] = B

for A_idx, A_entry in enumerate(A_list):
    file_prefix = file_prefixes[A_idx]

    for B in B_dict:
        nct_pipeline = ComputeMinimumControlEnergy(environment=environment, A=A_entry, states=states, B=B_dict[B],
                                                   control='minimum_fast', T=1,
                                                   file_prefix=file_prefix,
                                                   force_rerun=False, save_outputs=True, verbose=True)
        nct_pipeline.run()
