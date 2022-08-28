# %% import
import sys, os, platform

from src.imaging_derivs import DataMatrix
from src.pipelines import ComputeMinimumControlEnergy

import scipy as sp
from bct.algorithms.reference import randmio_und
from src.geomsurr import geomsurr
from tqdm import tqdm

# %% import workspace
os.environ["MY_PYTHON_WORKSPACE"] = 'ave_adj'
os.environ["WHICH_BRAIN_MAP"] = 'hist-g2'
# os.environ["WHICH_BRAIN_MAP"] = 'micro-g1'
# os.environ["WHICH_BRAIN_MAP"] = 'func-g1'
# os.environ["WHICH_BRAIN_MAP"] = 'myelin'
from setup_workspace import *

# %% rewire mean adjacency matrix with spatial constraints
D = sp.spatial.distance.pdist(environment.centroids, 'euclidean')
D = sp.spatial.distance.squareform(D)

wwp_file = 'average_adj_n-{0}_cthr-{1}_smap-{2}_null-mni-wwp'.format(load_average_sc.load_sc.df.shape[0],
                                                                   consist_thresh, which_brain_map)

wsp_file = 'average_adj_n-{0}_cthr-{1}_smap-{2}_null-mni-wsp'.format(load_average_sc.load_sc.df.shape[0],
                                                                   consist_thresh, which_brain_map)

if os.path.exists(os.path.join(environment.pipelinedir, 'minimum_control_energy', wwp_file+'.npy')) \
    and os.path.exists(os.path.join(environment.pipelinedir, 'minimum_control_energy', wsp_file+'.npy')):
    print('loading permuted A matrices')
    Wwp = np.load(os.path.join(environment.pipelinedir, 'minimum_control_energy', wwp_file+'.npy'))
    Wsp = np.load(os.path.join(environment.pipelinedir, 'minimum_control_energy', wsp_file+'.npy'))
    n_perms = Wwp.shape[2]
else:
    print('generating permuted A matrices')
    n_perms = 10000
    Wwp = np.zeros((n_parcels, n_parcels, n_perms))
    Wsp = np.zeros((n_parcels, n_parcels, n_perms))

    for i in tqdm(np.arange(n_perms)):
        np.random.seed(i)
        Wwp[:, :, i], Wsp[:, :, i], _ = geomsurr(W=A, D=D)

    np.save(os.path.join(environment.pipelinedir, 'minimum_control_energy', wwp_file), Wwp)
    np.save(os.path.join(environment.pipelinedir, 'minimum_control_energy', wsp_file), Wsp)


#%% compute energy
B = DataMatrix(data=np.eye(n_parcels), name='identity')
c = 1
T = 1

wwp_file = 'average_adj_n-{0}_cthr-{1}_smap-{2}_null-mni-wwp-E'.format(load_average_sc.load_sc.df.shape[0],
                                                                   consist_thresh, which_brain_map)

wsp_file = 'average_adj_n-{0}_cthr-{1}_smap-{2}_null-mni-wsp-E'.format(load_average_sc.load_sc.df.shape[0],
                                                                   consist_thresh, which_brain_map)

if os.path.exists(os.path.join(environment.pipelinedir, 'minimum_control_energy', wwp_file+'.npy')) \
    and os.path.exists(os.path.join(environment.pipelinedir, 'minimum_control_energy', wsp_file+'.npy')):
    print('loading permuted energy')
    E_wwp = np.load(os.path.join(environment.pipelinedir, 'minimum_control_energy', wwp_file+'.npy'))
    E_wsp = np.load(os.path.join(environment.pipelinedir, 'minimum_control_energy', wsp_file+'.npy'))
else:
    print('generating permuted energy')
    E_wwp = np.zeros((n_states, n_states, n_perms))
    E_wsp = np.zeros((n_states, n_states, n_perms))

    for i in tqdm(np.arange(n_perms)):
        # wwp
        file_prefix = 'average_adj_n-{0}_cthr-{1}_smap-{2}_null-mni-wwp-{3}_'.format(load_average_sc.load_sc.df.shape[0],
                                                                                        consist_thresh, which_brain_map, i)
        nct_pipeline = ComputeMinimumControlEnergy(environment=environment, A=Wwp[:, :, i], states=states, B=B,
                                                   control='minimum_fast', c=c, T=T,
                                                   file_prefix=file_prefix,
                                                   force_rerun=False, save_outputs=False, verbose=False)
        nct_pipeline.run()
        E_wwp[:, :, i] = nct_pipeline.E

        # n = 2
        # ds = 0.1
        # nct_pipeline.run_with_optimized_b(n=n, ds=ds)

        # wsp
        file_prefix = 'average_adj_n-{0}_cthr-{1}_smap-{2}_null-mni-wsp-{3}_'.format(load_average_sc.load_sc.df.shape[0],
                                                                                        consist_thresh, which_brain_map, i)
        nct_pipeline = ComputeMinimumControlEnergy(environment=environment, A=Wsp[:, :, i], states=states, B=B,
                                                   control='minimum_fast', c=c, T=T,
                                                   file_prefix=file_prefix,
                                                   force_rerun=False, save_outputs=False, verbose=False)
        nct_pipeline.run()
        E_wsp[:, :, i] = nct_pipeline.E

        # n = 2
        # ds = 0.1
        # nct_pipeline.run_with_optimized_b(n=n, ds=ds)

    # save
    np.save(os.path.join(environment.pipelinedir, 'minimum_control_energy', wwp_file), E_wwp)
    np.save(os.path.join(environment.pipelinedir, 'minimum_control_energy', wsp_file), E_wsp)
