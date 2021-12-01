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
os.environ["MY_PYTHON_WORKSPACE"] = 'ave_adj'
from setup_workspace import *

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
# A_list = [Wwp, Wsp, Wssp]
# file_prefixes = ['average_adj_n-{0}_cthr-{1}_smap-{2}_null-mni-wwp-{3}_'.format(load_average_sc.load_sc.df.shape[0],
#                                                                                 consist_thresh, which_brain_map,
#                                                                                 sge_task_id),
#                  'average_adj_n-{0}_cthr-{1}_smap-{2}_null-mni-wsp-{3}_'.format(load_average_sc.load_sc.df.shape[0],
#                                                                                 consist_thresh, which_brain_map,
#                                                                                 sge_task_id),
#                  'average_adj_n-{0}_cthr-{1}_smap-{2}_null-mni-wssp-{3}_'.format(load_average_sc.load_sc.df.shape[0],
#                                                                                  consist_thresh, which_brain_map,
#                                                                                  sge_task_id)]
A_list = [Wwp, ]
file_prefixes = ['average_adj_n-{0}_cthr-{1}_smap-{2}_null-mni-wwp-{3}_'.format(load_average_sc.load_sc.df.shape[0],
                                                                                 consist_thresh, which_brain_map,
                                                                                 sge_task_id), ]

B = DataMatrix(data=np.eye(n_parcels), name='identity')
c = 1
T = 1

for A_idx, A_entry in enumerate(A_list):
    file_prefix = file_prefixes[A_idx]

    nct_pipeline = ComputeMinimumControlEnergy(environment=environment, A=A_entry, states=states, B=B,
                                               control='minimum_fast', c=c, T=T,
                                               file_prefix=file_prefix,
                                               force_rerun=False, save_outputs=True, verbose=True)
    nct_pipeline.run()

    n = 2
    ds = 0.1
    nct_pipeline.run_with_optimized_b(n=n, ds=ds)
