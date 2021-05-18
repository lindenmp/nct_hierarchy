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
import scipy as sp
import numpy as np
from tqdm import tqdm

# %% Setup project environment
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='white', context='paper', font_scale=1)
if platform.system() == 'Linux':
    computer = 'cbica'
elif platform.system() == 'Darwin':
    computer = 'macbook'

    import matplotlib.font_manager as font_manager
    fontpath = '/Users/lindenmp/Library/Fonts/PublicSans-Thin.ttf'
    prop = font_manager.FontProperties(fname=fontpath)
    plt.rcParams['font.family'] = prop.get_name()
    plt.rcParams['svg.fonttype'] = 'none'

parc = 'schaefer'
n_parcels = 400
sc_edge_weight = 'streamlineCount'
environment = Environment(computer=computer, parc=parc, n_parcels=n_parcels, sc_edge_weight=sc_edge_weight)
environment.make_output_dirs()
environment.load_parc_data()

n_perms = 10000

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

# %% load mean brain maps
loaders_dict = {
    'ct': LoadCT(environment=environment, Subject=Subject),
    'cbf': LoadCBF(environment=environment, Subject=Subject)
}

load_average_bms = LoadAverageBrainMaps(loaders_dict=loaders_dict)
load_average_bms.run(return_descending=False)

# %% get control energy
file_prefix = 'average_adj_n-{0}_s-{1}_'.format(load_average_sc.load_sc.df.shape[0], spars_thresh)
n_subsamples = 0
save_outputs = True

# %% brain map null (spin test)
for key in load_average_bms.brain_maps:
    load_average_bms.brain_maps[key].shuffle_data(shuffle_indices=environment.spun_indices)

    E = np.zeros((compute_gradients.n_states, compute_gradients.n_states, n_perms))
    for sge_task_id in tqdm(np.arange(n_perms)):
        permuted_bm = DataVector(data=load_average_bms.brain_maps[key].data_shuf[:, sge_task_id].copy(),
                                 name='{0}-spin-{1}'.format(key, sge_task_id))

        nct_pipeline = ComputeMinimumControlEnergy(environment=environment, A=A,
                                                   states=compute_gradients.grad_bins, n_subsamples=n_subsamples,
                                                   control='minimum_fast', T=1, B=permuted_bm, file_prefix=file_prefix,
                                                   force_rerun=False, save_outputs=save_outputs, verbose=False)
        nct_pipeline.run()
        E[:, :, sge_task_id] = nct_pipeline.E

        # if output file exists, delete it
        # if os.path.isfile(os.path.join(nct_pipeline._output_dir(), '{0}E.npy'.format(nct_pipeline._get_file_prefix()))):
        #     os.remove(os.path.join(nct_pipeline._output_dir(), '{0}E.npy'.format(nct_pipeline._get_file_prefix())))

    # save compiled E
    my_list = nct_pipeline._get_file_prefix().split('_')
    my_list[-2] = ('-').join(my_list[-2].split('-')[:-1])
    out_prefix = ('_').join(my_list)
    np.save(os.path.join(nct_pipeline._output_dir(), '{0}E.npy'.format(out_prefix)), E)

# %% network null
A_list = ['wwp', 'wsp', 'wssp']
B_list = ['wb',] + list(load_average_bms.brain_maps.keys())

for A_entry in A_list:
    for B in B_list:
        E = np.zeros((compute_gradients.n_states, compute_gradients.n_states, n_perms))
        for sge_task_id in tqdm(np.arange(n_perms)):
            # rewire connectome
            D = sp.spatial.distance.pdist(environment.centroids, 'euclidean')
            D = sp.spatial.distance.squareform(D)
            octave.eval("rand('state',%i)" % sge_task_id)

            if A_entry == 'wwp':
                file_prefix = 'average_adj_n-{0}_s-{1}_null-mni-wwp-{2}_'.format(load_average_sc.load_sc.df.shape[0],
                                                                                 spars_thresh, sge_task_id)
                A_null, _, _ = octave.geomsurr(A, D, 3, 2, nout=3)
            elif A_entry == 'wsp':
                file_prefix = 'average_adj_n-{0}_s-{1}_null-mni-wsp-{2}_'.format(load_average_sc.load_sc.df.shape[0],
                                                                                 spars_thresh, sge_task_id)
                _, A_null, _ = octave.geomsurr(A, D, 3, 2, nout=3)
            elif A_entry == 'wssp':
                file_prefix = 'average_adj_n-{0}_s-{1}_null-mni-wssp-{2}_'.format(load_average_sc.load_sc.df.shape[0],
                                                                                  spars_thresh, sge_task_id)
                _, _, A_null = octave.geomsurr(A, D, 3, 2, nout=3)

            # compute energy
            if B == 'wb':
                nct_pipeline = ComputeMinimumControlEnergy(environment=environment, A=A_null,
                                                           states=compute_gradients.grad_bins, n_subsamples=n_subsamples,
                                                           control='minimum_fast', T=1, B=B, file_prefix=file_prefix,
                                                           force_rerun=False, save_outputs=save_outputs, verbose=False)
            else:
                nct_pipeline = ComputeMinimumControlEnergy(environment=environment, A=A_null,
                                                                   states=compute_gradients.grad_bins, n_subsamples=n_subsamples,
                                                                   control='minimum_fast', T=1, B=load_average_bms.brain_maps[B], file_prefix=file_prefix,
                                                                   force_rerun=False, save_outputs=save_outputs, verbose=False)

            nct_pipeline.run()
            E[:, :, sge_task_id] = nct_pipeline.E

            # if output file exists, delete it
            # if os.path.isfile(os.path.join(nct_pipeline._output_dir(), '{0}E.npy'.format(nct_pipeline._get_file_prefix()))):
            #     os.remove(os.path.join(nct_pipeline._output_dir(), '{0}E.npy'.format(nct_pipeline._get_file_prefix())))

        # save compiled E
        my_list = nct_pipeline._get_file_prefix().split('_')
        my_list[4] = ('-').join(my_list[4].split('-')[:-1])
        out_prefix = ('_').join(my_list)
        np.save(os.path.join(nct_pipeline._output_dir(), '{0}E.npy'.format(out_prefix)), E)
