# %%
import sys, platform
if platform.system() == 'Linux':
    sys.path.extend(['/cbica/home/parkesl/research_projects/pfactor_gradients'])
from pfactor_gradients.pipelines import ComputeMinimumControlEnergy, Regression
from pfactor_gradients.utils import rank_int
from pfactor_gradients.imaging_derivs import DataMatrix

from tqdm import tqdm

# %% import workspace
from setup_workspace_subj_adj import *

# %% plotting
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pfactor_gradients.plotting import set_plotting_params, my_reg_plot
set_plotting_params(format='svg')
figsize = 1.5

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
n_rand_splits = 50

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
                        n_rand_splits=n_rand_splits, force_rerun=False)
regression.run()
regression.run_perm()

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
                            n_rand_splits=n_rand_splits, force_rerun=False)
    regression.run()
    regression.run_perm()

# run with stratified
from sklearn.metrics import make_scorer
from sklearn.linear_model import Ridge
from pfactor_gradients.prediction import run_reg_scv, root_mean_squared_error

reg = Ridge()
scorer = make_scorer(root_mean_squared_error, greater_is_better=False)

accuracy, y_pred = run_reg_scv(X=X, y=y, c=c, reg=reg, scorer=scorer, n_splits=10, runpca=runpca)

# %% plotting
run_plots = False
if run_plots:
    # stratified
    f, ax = plt.subplots(1, 1, figsize=(figsize, figsize))
    my_reg_plot(np.sort(y)/12, y_pred/12, 'y_true (age in years)', 'y_pred (age in years)', ax, annotate='pearson')
    f.savefig(os.path.join(environment.figdir, 'prediction_y_pred_{0}'.format(y_name)), dpi=600, bbox_inches='tight',
              pad_inches=0.1)
    plt.close()

    # repeated cross val against nulls
    x1 = np.loadtxt(os.path.join(environment.pipelinedir, 'prediction',
                                'smap-{0}_X-energy-identity_y-{1}_c-{2}_alg-{3}_score-{4}_pca-{5}_accuracy_mean.txt' \
                                .format(which_brain_map, y_name, c_name, alg, score, runpca)))
    x2 = np.loadtxt(os.path.join(environment.pipelinedir, 'prediction',
                                'smap-{0}_X-energy-optimized_y-{1}_c-{2}_alg-{3}_score-{4}_pca-{5}_accuracy_mean.txt' \
                                .format(which_brain_map, y_name, c_name, alg, score, runpca)))

    x1_null = np.loadtxt(os.path.join(environment.pipelinedir, 'prediction',
                                'smap-{0}_X-energy-identity_y-{1}_c-{2}_alg-{3}_score-{4}_pca-{5}_accuracy_perm.txt' \
                                .format(which_brain_map, y_name, c_name, alg, score, runpca)))
    x2_null = np.loadtxt(os.path.join(environment.pipelinedir, 'prediction',
                                'smap-{0}_X-energy-optimized_y-{1}_c-{2}_alg-{3}_score-{4}_pca-{5}_accuracy_perm.txt' \
                                .format(which_brain_map, y_name, c_name, alg, score, runpca)))

    # cmap = sns.color_palette("Paired", as_cmap=False)
    cmap = sns.color_palette("pastel", as_cmap=False)
    # cmap = np.array([[255, 105, 97], [97, 168, 255]]) / 255
    # cmap = np.array([[124, 230, 199], [255, 169, 132]]) / 255

    f, ax = plt.subplots(1, 1, figsize=(figsize*0.5, figsize))

    sns.despine(left=True, bottom=True)
    ax.tick_params(pad=-2.5)

    # nulls (background)
    df_plot = pd.DataFrame(data=np.hstack((x1_null.reshape(-1, 1), x2_null.reshape(-1, 1))), columns=['identity', 'optimized'])
    df_plot = pd.melt(df_plot)
    df_plot["x"] = ""
    sns.violinplot(data=df_plot, ax=ax, x="x", y="value", hue="variable", inner=None, palette="pastel",
                   cut=2, linewidth=0.5, linecolor="k", split=True)

    for violin in ax.collections:
        violin.set_alpha(0.2)

    # observed (foreground)
    df_plot = pd.DataFrame(data=np.hstack((x1.reshape(-1, 1), x2.reshape(-1, 1))), columns=['identity', 'optimized'])
    df_plot = pd.melt(df_plot)
    df_plot["x"] = ""
    sns.violinplot(data=df_plot, ax=ax, x="x", y="value", hue="variable", inner=None, palette="pastel",
                   cut=2, linewidth=1.5, linecolor="k", split=True)

    n_violins = len(ax.collections)
    for violin in ax.collections[int(n_violins/2):]:
        violin.set_alpha(1)

    ax.legend_.remove()
    # handles, labels = ax.get_legend_handles_labels()
    # ax.legend(handles=handles[:2], labels=['identity', 'optimized'], title='', bbox_to_anchor=(1, 1.15), loc='upper right')

    # # add permutation p-values to plot
    # for i, B in enumerate(B_list):
    #     x_pos = my_list[i]
    #     y_pos = np.mean(df_pred.loc[df_pred['B'] == B, 'score'])
    #     if 'energy' in B:
    #         helper_func(x_pos, y_pos, p_vals_perm[i], True)
    #     else:
    #         helper_func(x_pos, y_pos, p_vals_perm[i], False)

    if score == 'rmse':
        ylabel = 'neg[RMSE]\n(higher = better)'
    elif score == 'corr':
        ylabel = 'correlation (y_true,y_pred)'
    elif score == 'r2':
        ylabel = 'R²'
    ax.set_xlabel('')
    ax.set_ylabel('')
    # ax.set_ylabel(ylabel)

    f.savefig(os.path.join(environment.figdir, 'prediction_{0}'.format(y_name)), dpi=600, bbox_inches='tight',
              pad_inches=0.1)
    plt.close()
