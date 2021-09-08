# %%
import sys, os, platform
import numpy as np
if platform.system() == 'Linux':
    sys.path.extend(['/cbica/home/parkesl/research_projects/pfactor_gradients'])
from pfactor_gradients.pipelines import ComputeMinimumControlEnergy, Regression
from pfactor_gradients.utils import rank_int
from pfactor_gradients.imaging_derivs import DataMatrix

from tqdm import tqdm

# %% import workspace
from setup_workspace_subj_adj import *

# %% parse input arguments
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-X_name", help="", dest="X_name", default='identity')
parser.add_argument("-y_name", help="", dest="y_name", default='ageAtScan1') # 'ageAtScan1' 'F1_Exec_Comp_Res_Accuracy'
parser.add_argument("-c_name", help="", dest="c_name", default='svm')
parser.add_argument("-alg", help="", dest="alg", default='rr')
parser.add_argument("-score", help="", dest="score", default='rmse')
parser.add_argument("-runpca", help="", dest="runpca", default='80%')

args = parser.parse_args()
print(args)
X_name = args.X_name
y_name = args.y_name
c_name = args.c_name
alg = args.alg
score = args.score
runpca = args.runpca

if type(runpca) == str and '%' not in runpca:
    runpca = int(runpca)

n_splits = 10
n_rand_splits = 100

# %% setup y and c
y = environment.df.loc[:, y_name].values

if np.any(np.isnan(y)):
    missing_data = np.isnan(y)
    print('filter {0} missing subjects...'.format(np.sum(missing_data)))
    y = y[~missing_data]
    # print('imputing missing data...')
    # y[np.isnan(y)] = np.nanmedian(y)

if c_name != None:
    if c_name == 'asvm':
        covs = ['ageAtScan1', 'sex', 'mprage_antsCT_vol_TBV', 'dti64MeanRelRMS']
    elif c_name == 'svm':
        covs = ['sex', 'mprage_antsCT_vol_TBV', 'dti64MeanRelRMS']
    elif c_name == 'vm':
        covs = ['mprage_antsCT_vol_TBV', 'dti64MeanRelRMS']

    c = environment.df.loc[:, covs]
    if 'sex' in covs:
        c['sex'] = c['sex'] - 1
    c = c.values

    # filter missing data
    try:
        c = c[~missing_data, :]
    except:
        pass
else:
    c = None

# %% get control energy
T = 1
B = DataMatrix(data=np.eye(n_parcels), name='identity')
E = np.zeros((n_states, n_states, n_subs))
if X_name == 'identity':
    E_opt = np.zeros((n_states, n_states, n_subs))

for i in tqdm(np.arange(n_subs)):
    file_prefix = '{0}_{1}_'.format(environment.df.index[i], which_brain_map)

    nct_pipeline = ComputeMinimumControlEnergy(environment=environment, A=load_sc.A[:, :, i], states=states, B=B,
                                               control='minimum_fast', T=T,
                                               file_prefix=file_prefix,
                                               force_rerun=False, save_outputs=True, verbose=False)

    if X_name == 'identity':
        n = 2
        ds = 0.1
        nct_pipeline.run_with_optimized_b(n=n, ds=ds)

        E[:, :, i] = nct_pipeline.E_opt[:, 0].reshape(n_states, n_states)
        E_opt[:, :, i] = nct_pipeline.E_opt[:, 1].reshape(n_states, n_states)
    else:
        nct_pipeline.run()
        E[:, :, i] = nct_pipeline.E

# %% prediction from energy
n_transitions = len(indices_upper[0])

# setup X
X = np.zeros((n_subs, n_transitions))
for i in np.arange(n_subs):
    e = E[:, :, i]
    e = rank_int(e)
    ed = e.transpose() - e
    X[i, :] = ed[indices_upper]

# filter missing data
try:
    X = X[~missing_data, :]
except:
    pass

# normalize energy
for i in np.arange(X.shape[1]):
    X[:, i] = rank_int(X[:, i])

regression = Regression(environment=environment, X=X, y=y, c=c, X_name='smap-{0}_X-energy-{1}'.format(which_brain_map, X_name),
                        y_name=y_name, c_name=c_name, alg=alg, score=score, n_splits=n_splits, runpca=runpca,
                        n_rand_splits=n_rand_splits, force_rerun=True)
regression.run()
# regression.run_perm()

# %% prediction from energy optimized
if X_name == 'identity':
    # setup X
    X = np.zeros((n_subs, n_transitions))
    for i in np.arange(n_subs):
        # e = E_opt[:, :, i][indices]
        # X[i, :] = e

        e = E_opt[:, :, i]
        e = rank_int(e)
        ed = e.transpose() - e
        X[i, :] = ed[indices_upper]

    # filter missing data
    try:
        X = X[~missing_data, :]
    except:
        pass

    # normalize energy
    for i in np.arange(X.shape[1]):
        X[:, i] = rank_int(X[:, i])

    regression = Regression(environment=environment, X=X, y=y, c=c, X_name='smap-{0}_X-energy-optimized'.format(which_brain_map),
                            y_name=y_name, c_name=c_name, alg=alg, score=score, n_splits=n_splits, runpca=runpca,
                            n_rand_splits=n_rand_splits, force_rerun=True)
    regression.run()
    # regression.run_perm()
