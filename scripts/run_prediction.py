# %%
import sys, os, platform
import numpy as np
if platform.system() == 'Linux':
    sys.path.extend(['/cbica/home/parkesl/research_projects/pfactor_gradients'])
from pfactor_gradients.pnc import Environment, Subject
from pfactor_gradients.routines import LoadSC, LoadCT, LoadRLFP, LoadCBF, LoadREHO, LoadALFF, LoadSA
from pfactor_gradients.pipelines import ComputeGradients
from pfactor_gradients.pipelines import Regression
from pfactor_gradients.utils import rank_int, get_states_from_gradient

# %% parse input arguments
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-X_name", help="", dest="X_name", default='wb')
parser.add_argument("-y_name", help="", dest="y_name", default='Overall_Psychopathology')
parser.add_argument("-c_name", help="", dest="c_name", default='asvm')
parser.add_argument("-alg", help="", dest="alg", default='rr')
parser.add_argument("-score", help="", dest="score", default='rmse')
parser.add_argument("-runpca", help="", dest="runpca", default='1%')

args = parser.parse_args()
print(args)
X_name = args.X_name
y_name = args.y_name
c_name = args.c_name
alg = args.alg
score = args.score
runpca = args.runpca

if type(runpca) and '%' not in runpca:
    runpca = int(runpca)

n_splits = 10
n_rand_splits = 100

# %% Setup project environment
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='white', context='talk', font_scale=1)
if platform.system() == 'Linux':
    computer = 'cbica'
    # sge_task_id = int(os.getenv("SGE_TASK_ID"))-1
elif platform.system() == 'Darwin':
    computer = 'macbook'
    # sge_task_id = 0

    import matplotlib.font_manager as font_manager
    fontpath = '/Users/lindenmp/Library/Fonts/PublicSans-Thin.ttf'
    prop = font_manager.FontProperties(fname=fontpath)
    plt.rcParams['font.family'] = prop.get_name()
    plt.rcParams['svg.fonttype'] = 'none'
# print(sge_task_id)

parc = 'schaefer'
n_parcels = 400
sc_edge_weight = 'streamlineCount'
environment = Environment(computer=computer, parc=parc, n_parcels=n_parcels, sc_edge_weight=sc_edge_weight)
environment.make_output_dirs()
environment.load_parc_data()

# filter subjects
filters = {'healthExcludev2': 0, 'psychoactiveMedPsychv2': 0,
           't1Exclude': 0, 'fsFinalExclude': 0,
           'b0ProtocolValidationStatus': 1, 'dti64ProtocolValidationStatus': 1, 'dti64Exclude': 0}
           # 'restProtocolValidationStatus': 1, 'restExclude': 0} # need to add these filters in if doing funcg1 below
environment.load_metadata(filters)

# %% get states
which_grad = 'histg2'

if which_grad == 'histg2':
    if computer == 'macbook':
        gradient = np.loadtxt('/Volumes/T7/research_data/BigBrainWarp/spaces/fsaverage/Hist_G2_Schaefer2018_400Parcels_17Networks.txt')
    elif computer == 'cbica':
        gradient = np.loadtxt('/cbica/home/parkesl/research_data/BigBrainWarp/spaces/fsaverage/Hist_G2_Schaefer2018_400Parcels_17Networks.txt')
    gradient = gradient * -1
elif which_grad == 'funcg1':
    # compute function gradient
    compute_gradients = ComputeGradients(environment=environment, Subject=Subject)
    compute_gradients.run()
    gradient = compute_gradients.gradients[:, 0]

n_bins = int(n_parcels/10)
states = get_states_from_gradient(gradient=gradient, n_bins=n_bins)
n_states = len(np.unique(states))

# %% Load sc data
load_sc = LoadSC(environment=environment, Subject=Subject)
load_sc.run()
# refilter environment due to LoadSC excluding on disconnected nodes
environment.df = load_sc.df.copy()
n_subs = environment.df.shape[0]
del load_sc

# %% load ct data
load_ct = LoadCT(environment=environment, Subject=Subject)
load_ct.run()

# refilter environment due to some missing FS subjects
environment.df = load_ct.df.copy()
n_subs = environment.df.shape[0]
del load_ct

# %%
y = environment.df.loc[:, y_name].values

if np.any(np.isnan(y)):
    y[np.isnan(y)] = np.nanmedian(y)

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

# %% prediction from brain maps (no NCT)
if X_name != 'wb' and 'mean' not in X_name and 'flip' not in X_name:
    if X_name == 'ct':
        loader = LoadCT(environment=environment, Subject=Subject)
    elif X_name == 'cbf':
        loader = LoadCBF(environment=environment, Subject=Subject)
    elif X_name == 'rlfp':
        loader = LoadRLFP(environment=environment, Subject=Subject)
    elif X_name == 'reho':
        loader = LoadREHO(environment=environment, Subject=Subject)
    elif X_name == 'alff':
        loader = LoadALFF(environment=environment, Subject=Subject)
    elif X_name == 'sa':
        loader = LoadSA(environment=environment, Subject=Subject)

    loader.run()
    X = loader.values

    # normalize
    for i in np.arange(X.shape[1]):
        X[:, i] = rank_int(X[:, i])

    # ct: 1% = 13 pcs
    # cbf: 1% = 9 pcs
    # reho: 1% = 10 pcs
    # alff: 1% = 8 pcs
    regression = Regression(environment=environment, X=X, y=y, c=c, X_name='{0}_{1}'.format(which_grad, X_name),
                            y_name=y_name, c_name=c_name, alg=alg, score=score, n_splits=n_splits, runpca=runpca,
                            n_rand_splits=n_rand_splits, force_rerun=False)
    regression.run()
    regression.run_perm()
else:
    pass

# %% prediction from energy
mask = ~np.eye(n_states, dtype=bool)
indices = np.where(mask)
n_transitions = len(indices[0])

# load energy
X = np.zeros((n_subs, n_transitions))
for i in np.arange(n_subs):
    subjid = environment.df.index[i]
    file = '{0}_{1}_ns-40-0_c-minimum_fast_T-1_B-{2}_E.npy'.format(subjid, which_grad, X_name)
    E = np.load(os.path.join(environment.pipelinedir, 'minimum_control_energy', file))
    X[i, :] = E[indices]

# normalize energy
for i in np.arange(X.shape[1]):
    X[:, i] = rank_int(X[:, i])

# wb: 1% = 37 pcs
# ct: 1% = 33 pcs
# cbf: 1% = 24 pcs
# reho: 1% = 21 pcs
# alff: 1% = 19 pcs
regression = Regression(environment=environment, X=X, y=y, c=c, X_name='{0}_energy-{1}'.format(which_grad, X_name),
                        y_name=y_name, c_name=c_name, alg=alg, score=score, n_splits=n_splits, runpca=runpca,
                        n_rand_splits=n_rand_splits, force_rerun=False)
regression.run()
regression.run_perm()

# # %% prediction from energy bottom-up
# indices = np.triu_indices(n_states, k=1)
# n_transitions = len(indices[0])
#
# # load energy
# X = np.zeros((n_subs, n_transitions))
# for i in np.arange(n_subs):
#     subjid = environment.df.index[i]
#     file = '{0}_ns-40-0_c-minimum_fast_T-1_B-{1}_E.npy'.format(subjid, X_name)
#     E = np.load(os.path.join(environment.pipelinedir, 'minimum_control_energy', file))
#     X[i, :] = E[indices]
#
# regression = Regression(environment=environment, X=X, y=y, c=c, X_name='energy-{0}-u'.format(X_name), y_name=y_name, c_name=c_name,
#                         alg=alg, score=score, n_splits=n_splits, runpca=runpca, n_rand_splits=n_rand_splits,
#                         force_rerun=False)
# regression.run()
# # regression.run_perm()
#
# # %% prediction from energy top-down
# indices = np.tril_indices(n_states, k=-1)
# n_transitions = len(indices[0])
#
# # load energy
# X = np.zeros((n_subs, n_transitions))
# for i in np.arange(n_subs):
#     subjid = environment.df.index[i]
#     file = '{0}_ns-40-0_c-minimum_fast_T-1_B-{1}_E.npy'.format(subjid, X_name)
#     E = np.load(os.path.join(environment.pipelinedir, 'minimum_control_energy', file))
#     X[i, :] = E[indices]
#
# regression = Regression(environment=environment, X=X, y=y, c=c, X_name='energy-{0}-l'.format(X_name), y_name=y_name, c_name=c_name,
#                         alg=alg, score=score, n_splits=n_splits, runpca=runpca, n_rand_splits=n_rand_splits,
#                         force_rerun=False)
# regression.run()
# # regression.run_perm()
