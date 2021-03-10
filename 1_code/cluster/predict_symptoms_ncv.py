import argparse

# Essentials
import os, sys, glob
import pandas as pd
import numpy as np
import copy
import json
from datetime import datetime

# Stats
import scipy as sp
from scipy import stats

# Sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR, LinearSVR
from sklearn.metrics import make_scorer, r2_score, mean_squared_error, mean_absolute_error
from sklearn.decomposition import PCA

# --------------------------------------------------------------------------------------------------------------------
# parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument("-x", help="IVs", dest="X_file", default=None)
parser.add_argument("-y", help="DVs", dest="y_file", default=None)
parser.add_argument("-metric", help="brain feature (e.g., ac)", dest="metric", default=None)
parser.add_argument("-pheno", help="psychopathology dimension", dest="pheno", default=None)
parser.add_argument("-seed", help="seed for shuffle_data", dest="seed", default=None)
parser.add_argument("-alg", help="estimator", dest="alg", default=None)
parser.add_argument("-score", help="score set order", dest="score", default=None)
parser.add_argument("-o", help="output directory", dest="outroot", default=None)

args = parser.parse_args()
print(args)
X_file = args.X_file
y_file = args.y_file
metric = args.metric
pheno = args.pheno
# seed = int(args.seed)
seed = int(os.environ['SGE_TASK_ID'])-1
alg = args.alg
score = args.score
outroot = args.outroot
# --------------------------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------------------------
# prediction functions
def corr_true_pred(y_true, y_pred):
    if type(y_true) == np.ndarray:
        y_true = y_true.flatten()
    if type(y_pred) == np.ndarray:
        y_pred = y_pred.flatten()
        
    r,p = sp.stats.pearsonr(y_true, y_pred)
    return r


def root_mean_squared_error(y_true, y_pred):
    mse = np.mean((y_true - y_pred)**2, axis=0)
    rmse = np.sqrt(mse)
    return rmse


def shuffle_data(X, y, seed = 0):
    np.random.seed(seed)
    idx = np.arange(y.shape[0])
    np.random.shuffle(idx)

    X_shuf = X.iloc[idx,:]
    y_shuf = y.iloc[idx]
    
    return X_shuf, y_shuf


def get_reg(num_params = 25):
    regs = {'rr': Ridge(),
            'lr': Lasso(),
            'krr_lin': KernelRidge(kernel='linear'),
            'krr_rbf': KernelRidge(kernel='rbf'),
            'svr_lin': SVR(kernel='linear'),
            'svr_rbf': SVR(kernel='rbf')
            }
    
    # From the sklearn docs, gamma defaults to 1/n_features.
    alpha_range = [1, -1]
    gamma_range = [-2, -4]
    param_grids = {'rr': {'reg__alpha': np.logspace(alpha_range[0], alpha_range[1], num_params)},
                    'lr': {'reg__alpha': np.logspace(alpha_range[0], alpha_range[1], num_params)},
                    'krr_lin': {'reg__alpha': np.logspace(alpha_range[0], alpha_range[1], num_params)},
                    # 'krr_rbf': {'reg__alpha': np.logspace(alpha_range[0], alpha_range[1], num_params)},
                    'krr_rbf': {'reg__alpha': np.logspace(alpha_range[0], alpha_range[1], num_params), 'reg__gamma': np.logspace(gamma_range[0], gamma_range[1], num_params)},
                    'svr_lin': {'reg__C': np.logspace(0, 4, num_params)},
                    'svr_rbf': {'reg__C': np.logspace(0, 4, num_params), 'reg__gamma': np.logspace(0, -3, num_params)}
                    }
    
    return regs, param_grids


def run_reg_ncv(X, y, reg, param_grid, n_splits = 10, scoring = 'r2'):
    
    pipe = Pipeline(steps=[('standardize', StandardScaler()),
                            ('reg', reg)])
                            # ('PCA', PCA(n_components = 5, svd_solver = 'full')),
    
    inner_cv = KFold(n_splits = n_splits, shuffle = False, random_state = None)
    outer_cv = KFold(n_splits = n_splits, shuffle = False, random_state = None)
    
    # if scoring is a dictionary then we run GridSearchCV with multiple scoring metrics and refit using the first one in the dict
    if type(scoring) == dict: grid = GridSearchCV(pipe, param_grid, cv = inner_cv, scoring = scoring, refit = list(scoring.keys())[0])
    else: grid = GridSearchCV(pipe, param_grid, cv = inner_cv, scoring = scoring)
    
    grid.fit(X, y);
    nested_score = cross_val_score(grid, X, y, cv = outer_cv)
    
    return grid, nested_score


def reg_ncv_wrapper(X, y, alg = 'krr_rbf', seed = 0, scoring = 'r2'):
        
    # get regression estimator
    regs, param_grids = get_reg()
    
    # run nested cv w/ shuffle
    X_shuf, y_shuf = shuffle_data(X = X, y = y, seed = seed)
    grid, nested_score = run_reg_ncv(X = X_shuf, y = y_shuf, reg = regs[alg], param_grid = param_grids[alg], scoring = scoring)
    
    best_params = grid.best_params_

    if type(scoring) == dict:
        best_scores = dict()
        for key in scoring.keys():
            best_scores[key] = grid.cv_results_['mean_test_'+key][grid.best_index_]
    else:
        best_scores = grid.best_score_
    
    return best_params, best_scores, nested_score


# --------------------------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------------------------
# start timer
start = datetime.now()

# inputs
X = pd.read_csv(X_file)
X.set_index(['bblid', 'scanid'], inplace = True)
X = X.filter(regex = metric+'_')

y = pd.read_csv(y_file)
y.set_index(['bblid', 'scanid'], inplace = True)
y = y.loc[:,pheno]

# outdir
outdir = os.path.join(outroot, 'split_' + str(seed), alg + '_' + score + '_' + metric + '_' + pheno)
if not os.path.exists(outdir): os.makedirs(outdir);
# --------------------------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------------------------
# set scorer
my_scorer_corr = make_scorer(corr_true_pred, greater_is_better = True)
my_scorer_rmse = make_scorer(root_mean_squared_error, greater_is_better = False)

if score == 'r2':
    scoring = {'r2': 'r2', 'mse': 'neg_mean_squared_error', 'rmse': my_scorer_rmse, 'mae': 'neg_mean_absolute_error', 'corr': my_scorer_corr}
elif score == 'corr':
    scoring = {'corr': my_scorer_corr, 'r2': 'r2', 'mse': 'neg_mean_squared_error', 'rmse': my_scorer_rmse, 'mae': 'neg_mean_absolute_error'}
elif score == 'mse':
    scoring = {'mse': 'neg_mean_squared_error', 'r2': 'r2', 'rmse': my_scorer_rmse, 'mae': 'neg_mean_absolute_error', 'corr': my_scorer_corr}
elif score == 'rmse':
    scoring = {'rmse': my_scorer_rmse, 'r2': 'r2', 'mse': 'neg_mean_squared_error', 'mae': 'neg_mean_absolute_error', 'corr': my_scorer_corr}
elif score == 'mae':
    scoring = {'mae': 'neg_mean_absolute_error', 'r2': 'r2', 'mse': 'neg_mean_squared_error', 'rmse': my_scorer_rmse, 'corr': my_scorer_corr}

# prediction
best_params, best_scores, nested_score = reg_ncv_wrapper(X = X, y = y, alg = alg, seed = seed, scoring = scoring)

# stop timer
stop = datetime.now()
# --------------------------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------------------------
# outputs
json_data = json.dumps(best_params)
f = open(os.path.join(outdir,'best_params.json'),'w')
f.write(json_data)
f.close()

json_data = json.dumps(best_scores)
f = open(os.path.join(outdir,'best_scores.json'),'w')
f.write(json_data)
f.close()

np.savetxt(os.path.join(outdir,'nested_score.csv'), nested_score, delimiter=',')

# runtime
run_time = stop - start
np.savetxt(os.path.join(outdir,'run_time_seconds.txt'), np.array([run_time.seconds]))
# --------------------------------------------------------------------------------------------------------------------

print('Finished!')
