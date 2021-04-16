# %%
import sys, os, platform
import numpy as np
if platform.system() == 'Linux':
    sys.path.extend(['/cbica/home/parkesl/research_projects/pfactor_gradients'])
from pfactor_gradients.pnc import Environment, Subject
from pfactor_gradients.routines import LoadSC, LoadCT, LoadRLFP, LoadCBF, LoadREHO, LoadALFF
from pfactor_gradients.pipelines import ComputeGradients
from pfactor_gradients.pipelines import Regression

# %% Setup project environment
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='white', context='talk', font_scale=1)
if platform.system() == 'Linux':
    computer = 'cbica'
    sge_task_id = int(os.getenv("SGE_TASK_ID"))-1
elif platform.system() == 'Darwin':
    computer = 'macbook'
    sge_task_id = 0

    import matplotlib.font_manager as font_manager
    fontpath = '/Users/lindenmp/Library/Fonts/PublicSans-Thin.ttf'
    prop = font_manager.FontProperties(fname=fontpath)
    plt.rcParams['font.family'] = prop.get_name()
    plt.rcParams['svg.fonttype'] = 'none'
print(sge_task_id)

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

# %% prediction settings
def get_prediction_pipeline(sge_task_id=np.nan):
    X_name = 'wb'
    y_name = 'Overall_Psychopathology'
    c_name = 'asvm'
    covs = ['ageAtScan1', 'sex', 'mprage_antsCT_vol_TBV', 'dti64MeanRelRMS']
    alg = 'rr'
    score = 'rmse'
    n_splits = 10
    runpca = '1%'
    n_rand_splits = 100

    if sge_task_id == 0:
        X_name = 'wb'
    elif sge_task_id == 1:
        X_name = 'ct'
    elif sge_task_id == 2:
        X_name = 'cbf'
    elif sge_task_id == 3:
        X_name = 'reho'
    elif sge_task_id == 4:
        X_name = 'alff'

    elif sge_task_id == 5:
        X_name = 'wb'
        score = 'corr'
    elif sge_task_id == 6:
        X_name = 'ct'
        score = 'corr'
    elif sge_task_id == 7:
        X_name = 'cbf'
        score = 'corr'
    elif sge_task_id == 8:
        X_name = 'reho'
        score = 'corr'
    elif sge_task_id == 9:
        X_name = 'alff'
        score = 'corr'

    elif sge_task_id == 10:
        X_name = 'wb'
        runpca = 25
    elif sge_task_id == 11:
        X_name = 'ct'
        runpca = 25
    elif sge_task_id == 12:
        X_name = 'cbf'
        runpca = 25
    elif sge_task_id == 13:
        X_name = 'reho'
        runpca = 25
    elif sge_task_id == 14:
        X_name = 'alff'
        runpca = 25

    elif sge_task_id == 15:
        X_name = 'wb'
        score = 'corr'
        runpca = 25
    elif sge_task_id == 16:
        X_name = 'ct'
        score = 'corr'
        runpca = 25
    elif sge_task_id == 17:
        X_name = 'cbf'
        score = 'corr'
        runpca = 25
    elif sge_task_id == 18:
        X_name = 'reho'
        score = 'corr'
        runpca = 25
    elif sge_task_id == 19:
        X_name = 'alff'
        score = 'corr'
        runpca = 25

    return X_name, y_name, c_name, covs, alg, score, n_splits, runpca, n_rand_splits

X_name, y_name, c_name, covs, alg, score, n_splits, runpca, n_rand_splits = get_prediction_pipeline(
    sge_task_id=sge_task_id)

y = environment.df.loc[:, y_name].values
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
