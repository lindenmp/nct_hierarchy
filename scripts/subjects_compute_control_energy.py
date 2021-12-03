# %%
import sys, os, platform
if platform.system() == 'Linux':
    sys.path.extend(['/cbica/home/parkesl/research_projects/pfactor_gradients'])
from pfactor_gradients.pipelines import ComputeMinimumControlEnergy
from pfactor_gradients.imaging_derivs import DataMatrix

# %% import workspace
os.environ["MY_PYTHON_WORKSPACE"] = 'subj_adj'
os.environ["WHICH_BRAIN_MAP"] = 'hist-g2'
# os.environ["WHICH_BRAIN_MAP"] = 'func-g1'
from setup_workspace import *

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

# %% compute minimum energy
file_prefix = '{0}_{1}_'.format(environment.df.index[0], which_brain_map)
B = DataMatrix(data=np.eye(n_parcels), name='identity')
c = 1
T = 1

nct_pipeline = ComputeMinimumControlEnergy(environment=environment, A=A, states=states, B=B,
                                           control='minimum_fast', c=c, T=T,
                                           file_prefix=file_prefix,
                                           force_rerun=False, save_outputs=True, verbose=True)
nct_pipeline.run()

n = 2
ds = 0.1
nct_pipeline.run_with_optimized_b(n=n, ds=ds)
