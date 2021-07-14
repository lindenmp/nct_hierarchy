# %%
import sys, os, platform
if platform.system() == 'Linux':
    sys.path.extend(['/cbica/home/parkesl/research_projects/pfactor_gradients'])
from pfactor_gradients.routines import LoadCT, LoadSA
from pfactor_gradients.pipelines import ComputeMinimumControlEnergy
from pfactor_gradients.imaging_derivs import DataMatrix

# %% import workspace
from setup_workspace_subj_adj import *

# %% Setup project environment
if platform.system() == 'Linux':
    sge_task_id = int(os.getenv("SGE_TASK_ID")) - 1
elif platform.system() == 'Darwin':
    sge_task_id = 0
print(sge_task_id)

# %% get subject A matrix out
A = load_sc.A[:, :, sge_task_id].copy()
print(load_sc.df.index[sge_task_id])

environment.df = environment.df.iloc[sge_task_id, :].to_frame().transpose()
print(environment.df.index[0])

# %% load subj's brain maps
loaders_dict = {
    'ct': LoadCT(environment=environment, Subject=Subject),
    'sa': LoadSA(environment=environment, Subject=Subject)
}

for key in loaders_dict:
    loaders_dict[key].run()

# %% compute minimum energy
T = 1
file_prefix = '{0}_{1}_'.format(environment.df.index[0], which_brain_map)

B_dict = dict()

B = DataMatrix(data=np.eye(n_parcels), name='identity')
B_dict[B.name] = B

for key in loaders_dict:
    try:
        B = DataMatrix(data=np.zeros((n_parcels, n_parcels)), name=key)
        B.data[np.eye(n_parcels) == 1] = 1 + loaders_dict[key].values[0, :]
        B_dict[B.name] = B
    except IndexError:
        pass

for B in B_dict:
    nct_pipeline = ComputeMinimumControlEnergy(environment=environment, A=A, states=states, B=B_dict[B],
                                               control='minimum_fast', T=T,
                                               file_prefix=file_prefix,
                                               force_rerun=False, save_outputs=True, verbose=True)
    nct_pipeline.run()

# %% compute minimum energy, control set: optimized B
nct_pipeline = ComputeMinimumControlEnergy(environment=environment, A=A, states=states, B=B_dict['identity'],
                                           control='minimum_fast', T=T,
                                           file_prefix=file_prefix,
                                           force_rerun=False, save_outputs=True, verbose=True)
nct_pipeline.run_with_optimized_b()
