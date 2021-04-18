# %%
import sys, os, platform
import numpy as np
if platform.system() == 'Linux':
    sys.path.extend(['/cbica/home/parkesl/research_projects/pfactor_gradients'])
from pfactor_gradients.pnc import Environment, Subject
from pfactor_gradients.routines import LoadSC, LoadCT, LoadRLFP, LoadCBF, LoadREHO, LoadALFF
from pfactor_gradients.pipelines import ComputeGradients
from pfactor_gradients.pipelines import Regression

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

# %% get clustered gradients
filters = {'healthExcludev2': 0, 't1Exclude': 0,
           'b0ProtocolValidationStatus': 1, 'dti64ProtocolValidationStatus': 1, 'dti64Exclude': 0,
           'psychoactiveMedPsychv2': 0, 'restProtocolValidationStatus': 1, 'restExclude': 0}
environment.load_metadata(filters)
compute_gradients = ComputeGradients(environment=environment, Subject=Subject)
compute_gradients.run()

# %% Load sc data
load_sc = LoadSC(environment=environment, Subject=Subject)
load_sc.run()
# refilter environment due to LoadSC excluding on disconnected nodes
environment.df = load_sc.df.copy()
n_subs = environment.df.shape[0]

y = environment.df.loc[:, y_name].values
if c_name == 'asvm':
    covs = ['ageAtScan1', 'sex', 'mprage_antsCT_vol_TBV', 'dti64MeanRelRMS']
c = environment.df.loc[:, covs]
c['sex'] = c['sex'] - 1
c = c.values

# %% prediction from brain maps (no NCT)
if X_name != 'wb':
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

    loader.run()
    X = loader.values

    # ct: 1% = 13 pcs
    # cbf: 1% = 9 pcs
    # reho: 1% = 10 pcs
    # alff: 1% = 8 pcs
    regression = Regression(environment=environment, X=X, y=y, c=c, X_name=X_name, y_name=y_name, c_name=c_name,
                            alg=alg, score=score, n_splits=n_splits, runpca=runpca, n_rand_splits=n_rand_splits,
                            force_rerun=False)
    regression.run()
    regression.run_perm()
else:
    pass

# %% prediction from energy
n_states = len(np.unique(compute_gradients.grad_bins))
mask = ~np.eye(n_states, dtype=bool)
indices = np.where(mask)
# indices = np.triu_indices(n_states, k=1)
n_transitions = len(indices[0])

# load energy
X = np.zeros((n_subs, n_transitions))
for i in np.arange(n_subs):
    subjid = environment.df.index[i]
    file = '{0}_ns-40-0_c-minimum_fast_T-1_B-{1}_E.npy'.format(subjid, X_name)
    E = np.load(os.path.join(environment.pipelinedir, 'minimum_control_energy', file))
    X[i, :] = E[indices]

# wb: 1% = 37 pcs
# ct: 1% = 33 pcs
# cbf: 1% = 24 pcs
# reho: 1% = 21 pcs
# alff: 1% = 19 pcs
regression = Regression(environment=environment, X=X, y=y, c=c, X_name='energy-{0}'.format(X_name), y_name=y_name, c_name=c_name,
                        alg=alg, score=score, n_splits=n_splits, runpca=runpca, n_rand_splits=n_rand_splits,
                        force_rerun=False)
regression.run()
regression.run_perm()
