# %% import
import sys, os, platform

from src.pipelines import ComputeMinimumControlEnergy
from src.utils import get_bootstrap_indices

from tqdm import tqdm

# %% import workspace
os.environ["MY_PYTHON_WORKSPACE"] = 'ave_adj'
os.environ["WHICH_BRAIN_MAP"] = 'hist-g2'
# os.environ["WHICH_BRAIN_MAP"] = 'micro-g1'
# os.environ["WHICH_BRAIN_MAP"] = 'func-g1'
# os.environ["WHICH_BRAIN_MAP"] = 'myelin'
from setup_workspace import *

# %% generate bootstrapped energy
B = DataMatrix(data=np.eye(n_parcels), name='identity')
c = 1
T = 1
frac = 0.5

A_file = 'average_adj_n-{0}_cthr-{1}_smap-{2}_bootstrapped_frac{3}_Am'.format(load_average_sc.load_sc.df.shape[0],
                                                                              consist_thresh, which_brain_map,
                                                                              str(frac).replace('.', ''))

E_file = 'average_adj_n-{0}_cthr-{1}_smap-{2}_bootstrapped_frac{3}_E'.format(load_average_sc.load_sc.df.shape[0],
                                                                             consist_thresh, which_brain_map,
                                                                             str(frac).replace('.', ''))

if os.path.exists(os.path.join(environment.pipelinedir, 'minimum_control_energy', E_file+'.npy')):
    print('loading bootstrapped energy')
    e_bs = np.load(os.path.join(environment.pipelinedir, 'minimum_control_energy', E_file+'.npy'))
    n_samples = e_bs.shape[2]
else:
    print('generating bootstrapped energy')
    n_samples = 1000
    Am_bs = np.zeros((n_parcels, n_parcels, n_samples))
    e_bs = np.zeros((n_states, n_states, n_samples))

    bootstrap_indices = get_bootstrap_indices(d_size=n_subs, frac=frac, n_samples=n_samples)

    for i in tqdm(np.arange(n_samples)):
        file_prefix = 'average_adj_n-{0}_cthr-{1}_smap-{2}_strap-{3}_'.format(load_average_sc.load_sc.df.shape[0],
                                                                              consist_thresh, which_brain_map, i)

        load_sc_strap = LoadSC(environment=environment, Subject=Subject)
        load_sc_strap.df = load_sc.df.iloc[bootstrap_indices[i, :], :].copy()
        load_sc_strap.A = load_sc.A[:, :, bootstrap_indices[i, :]].copy()

        load_average_sc_strap = LoadAverageSC(load_sc=load_sc_strap, consist_thresh=consist_thresh, verbose=False)
        load_average_sc_strap.run()
        Am_bs[:, :, i] = load_average_sc_strap.A

        # get bootstrapped energy
        nct_pipeline = ComputeMinimumControlEnergy(environment=environment, A=load_average_sc_strap.A, states=states, B=B,
                                                   control='minimum_fast', c=c, T=T,
                                                   file_prefix=file_prefix,
                                                   force_rerun=False, save_outputs=False, verbose=False)
        nct_pipeline.run()
        e_bs[:, :, i] = nct_pipeline.E

    # save
    np.save(os.path.join(environment.pipelinedir, 'minimum_control_energy', A_file), Am_bs)
    np.save(os.path.join(environment.pipelinedir, 'minimum_control_energy', E_file), e_bs)
