# %% import
import sys, os, platform
if platform.system() == 'Linux':
    sge_task_id = int(os.getenv("SGE_TASK_ID"))-1
    sys.path.extend(['/cbica/home/parkesl/research_projects/nct_hierarchy'])
elif platform.system() == 'Darwin':
    sge_task_id = 0
print(sge_task_id)

from src.pipelines import ComputeMinimumControlEnergy
from src.utils import get_bootstrap_indices

# %% import workspace
os.environ["MY_PYTHON_WORKSPACE"] = 'ave_adj'
os.environ["WHICH_BRAIN_MAP"] = 'hist-g2'
from setup_workspace import *

# %% bootstrap
n_samples = 1000
bootstrap_indices = get_bootstrap_indices(d_size=n_subs, frac=0.5, n_samples=n_samples)

file_prefix = 'average_adj_n-{0}_cthr-{1}_smap-{2}_strap-{3}_'.format(load_average_sc.load_sc.df.shape[0],
                                                                      consist_thresh, which_brain_map, sge_task_id)

load_sc_strap = LoadSC(environment=environment, Subject=Subject)
load_sc_strap.df = load_sc.df.iloc[bootstrap_indices[sge_task_id, :], :].copy()
load_sc_strap.A = load_sc.A[:, :, bootstrap_indices[sge_task_id, :]].copy()

load_average_sc_strap = LoadAverageSC(load_sc=load_sc_strap, consist_thresh=consist_thresh, verbose=True)
load_average_sc_strap.run()

# get bootstrapped energy
B = DataMatrix(data=np.eye(n_parcels), name='identity')
c = 1
T = 1

nct_pipeline = ComputeMinimumControlEnergy(environment=environment, A=load_average_sc_strap.A, states=states, B=B,
                                           control='minimum_fast', c=c, T=T,
                                           file_prefix=file_prefix,
                                           force_rerun=False, save_outputs=True, verbose=True)
n = 2
ds = 0.1
nct_pipeline.run_with_optimized_b(n=n, ds=ds)
