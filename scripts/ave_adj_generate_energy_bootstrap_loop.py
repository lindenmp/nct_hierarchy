# %% import
import sys, os, platform

from src.imaging_derivs import DataMatrix
from src.pipelines import ComputeMinimumControlEnergy
from src.utils import get_bootstrap_indices

from tqdm import tqdm

# %% import workspace
os.environ["MY_PYTHON_WORKSPACE"] = 'ave_adj'
os.environ["WHICH_BRAIN_MAP"] = 'hist-g2'
from setup_workspace import *

# %% generate bootstrapped energy
B = DataMatrix(data=np.eye(n_parcels), name='identity')
c = 1
T = 1

E_file = 'average_adj_n-{0}_cthr-{1}_smap-{2}_bootstrapped_E'.format(load_average_sc.load_sc.df.shape[0],
                                                                     consist_thresh, which_brain_map)

if os.path.exists(os.path.join(environment.pipelinedir, 'minimum_control_energy', E_file+'.npy')):
    print('loading bootstrapped energy')
    e_bs = np.load(os.path.join(environment.pipelinedir, 'minimum_control_energy', E_file+'.npy'))
    n_samples = e_bs.shape[2]
else:
    print('generating bootstrapped energy')
    n_samples = 10000
    e_bs = np.zeros((n_states, n_states, n_samples))

    bootstrap_indices = get_bootstrap_indices(d_size=n_subs, frac=1, n_samples=n_samples)

    for i in tqdm(np.arange(n_samples)):
        file_prefix = 'average_adj_n-{0}_cthr-{1}_smap-{2}_strap-{3}_'.format(load_average_sc.load_sc.df.shape[0],
                                                                              consist_thresh, which_brain_map, i)

        load_sc_strap = LoadSC(environment=environment, Subject=Subject)
        load_sc_strap.df = load_sc.df.iloc[bootstrap_indices[i, :], :].copy()
        load_sc_strap.A = load_sc.A[:, :, bootstrap_indices[i, :]].copy()

        load_average_sc_strap = LoadAverageSC(load_sc=load_sc_strap, consist_thresh=consist_thresh, verbose=False)
        load_average_sc_strap.run()

        # get bootstrapped energy
        nct_pipeline = ComputeMinimumControlEnergy(environment=environment, A=load_average_sc_strap.A, states=states, B=B,
                                                   control='minimum_fast', c=c, T=T,
                                                   file_prefix=file_prefix,
                                                   force_rerun=False, save_outputs=False, verbose=False)
        nct_pipeline.run()
        e_bs[:, :, i] = nct_pipeline.E

    # save
    np.save(os.path.join(environment.pipelinedir, 'minimum_control_energy', E_file), e_bs)
