# %%
import sre_parse
import sys, os, platform
from pfactor_gradients.pnc import Environment
from pfactor_gradients.utils import get_exact_p, get_null_p, get_fdr_p
import numpy as np
import pandas as pd
import scipy as sp

# %% Plotting
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='whitegrid', context='paper', font_scale=1)
import matplotlib.font_manager as font_manager
fontpath = '/System/Library/Fonts/HelveticaNeue.ttc'
prop = font_manager.FontProperties(fname=fontpath)
prop.set_weight = 'thin'
plt.rcParams['font.family'] = prop.get_family()
plt.rcParams['font.sans-serif'] = prop.get_name()
# plt.rcParams['font.weight'] = 'thin'
plt.rcParams['svg.fonttype'] = 'none'

# %% Setup project environment
computer = 'macbook'
parc = 'schaefer'
n_parcels = 400
sc_edge_weight = 'streamlineCount'
environment = Environment(computer=computer, parc=parc, n_parcels=n_parcels, sc_edge_weight=sc_edge_weight)
environment.make_output_dirs()
# environment.load_parc_data()

# %% load prediction results
y_name = 'F1_Exec_Comp_Res_Accuracy' # 'F1_Exec_Comp_Res_Accuracy' 'F1_Complex_Reasoning_Efficiency' 'F3_Executive_Efficiency'
y_label = 'Cognition'
c_name = 'asvm'
alg = 'rr'
score = 'rmse'
runpca = '80%'

# load unweighted energy
df_prediction = pd.DataFrame(columns=['score', 'B', 'energy'])
file_prefix = '{0}-{1}-{2}_alg-{3}_score-{4}_pca-{5}_'.format('energy-wb', y_name, c_name,
                                                              alg, score, runpca)
accuracy_mean = np.loadtxt(os.path.join(environment.pipelinedir, 'prediction',
                                        '{0}accuracy_mean.txt'.format(file_prefix)))
df_tmp = pd.DataFrame(columns=['score', 'B', 'energy'])
df_tmp['score'] = accuracy_mean
df_tmp['B'] = 'wb'
df_tmp['energy'] = 'yes'
df_prediction = pd.concat((df_prediction, df_tmp), axis=0)

# load unweighted energy null
df_prediction_null = pd.DataFrame(columns=['score', 'B', 'energy'])
accuracy_perm = np.loadtxt(os.path.join(environment.pipelinedir, 'prediction',
                                        '{0}accuracy_perm.txt'.format(file_prefix)))
df_tmp = pd.DataFrame(columns=['score', 'B', 'energy'])
df_tmp['score'] = accuracy_perm
df_tmp['B'] = 'wb'
df_tmp['energy'] = 'yes'
df_prediction_null = pd.concat((df_prediction_null, df_tmp), axis=0)

# load prediction from brain maps and weighted energgy
B_list = ['ct', 'cbf', 'energy-ct', 'energy-cbf']
# B_list = ['ct', 'cbf', 'energy-ct-u', 'energy-cbf-u']
# B_list = ['ct', 'cbf', 'energy-ct-l', 'energy-cbf-l']
n_B = len(B_list)
p_vals = np.zeros(n_B)

for i in np.arange(n_B):
    # observed prediction performance
    file_prefix = '{0}-{1}-{2}_alg-{3}_score-{4}_pca-{5}_'.format(B_list[i], y_name, c_name,
                                                                  alg, score, runpca)
    accuracy_mean = np.loadtxt(os.path.join(environment.pipelinedir, 'prediction',
                                            '{0}accuracy_mean.txt'.format(file_prefix)))
    df_tmp = pd.DataFrame(columns=['score', 'B', 'energy'])
    df_tmp['score'] = accuracy_mean
    if len(B_list[i].split('-')) >= 2:
        df_tmp['B'] = B_list[i].split('-')[1]
        df_tmp['energy'] = 'yes'
    elif len(B_list[i].split('-')) == 1:
        df_tmp['B'] = B_list[i].split('-')[0]
        df_tmp['energy'] = 'no'
    df_prediction = pd.concat((df_prediction, df_tmp), axis=0)

    # null prediction performance
    accuracy_perm = np.loadtxt(os.path.join(environment.pipelinedir, 'prediction',
                                            '{0}accuracy_perm.txt'.format(file_prefix)))
    df_tmp = pd.DataFrame(columns=['score', 'B', 'energy'])
    df_tmp['score'] = accuracy_perm
    if len(B_list[i].split('-')) == 2:
        df_tmp['B'] = B_list[i].split('-')[1]
        df_tmp['energy'] = 'yes'
    elif len(B_list[i].split('-')) == 1:
        df_tmp['B'] = B_list[i].split('-')[0]
        df_tmp['energy'] = 'no'
    df_prediction_null = pd.concat((df_prediction_null, df_tmp), axis=0)

# %% compute statistical comparisons and print them to python console
B = 'wb'
print('{0}'.format(B))
print('\tpermutation test')
x = np.mean(df_prediction.loc[np.logical_and(df_prediction['B'] == B,
                                             df_prediction['energy'] == 'yes'), 'score'])
null = df_prediction_null.loc[np.logical_and(df_prediction_null['B'] == B,
                                             df_prediction_null['energy'] == 'yes'), 'score']
print('\t\tenergy p = {0}'.format(get_null_p(x, null)))

for B in ['ct', 'cbf']:
    print('{0}'.format(B))

    print('\tweighted energy vs unweighted energy')
    x = df_prediction.loc[np.logical_and(df_prediction['B'] == B, df_prediction['energy'] == 'yes'), 'score']
    y = df_prediction.loc[np.logical_and(df_prediction['B'] == 'wb', df_prediction['energy'] == 'yes'), 'score']
    print('\t\texact p = {0}'.format(get_exact_p(x, y)))
    print('\t\t{0}'.format(sp.stats.wilcoxon(x, y)))

    print('\tenergy vs no energy')
    x = df_prediction.loc[np.logical_and(df_prediction['B'] == B, df_prediction['energy'] == 'yes'), 'score']
    y = df_prediction.loc[np.logical_and(df_prediction['B'] == B, df_prediction['energy'] == 'no'), 'score']
    print('\t\texact p = {0}'.format(get_exact_p(x, y)))
    print('\t\t{0}'.format(sp.stats.wilcoxon(x, y)))

    print('\tpermutation test')
    x = np.mean(df_prediction.loc[np.logical_and(df_prediction['B'] == B,
                                                 df_prediction['energy'] == 'yes'), 'score'])
    null = df_prediction_null.loc[np.logical_and(df_prediction_null['B'] == B,
                                                 df_prediction_null['energy'] == 'yes'), 'score']
    print('\t\tenergy p = {0}'.format(get_null_p(x, null)))

    x = np.mean(df_prediction.loc[np.logical_and(df_prediction['B'] == B,
                                                 df_prediction['energy'] == 'no'), 'score'])
    null = df_prediction_null.loc[np.logical_and(df_prediction_null['B'] == B,
                                                 df_prediction_null['energy'] == 'no'), 'score']
    print('\t\tno energy p = {0}'.format(get_null_p(x, null)))

# %% plot
# cmap = sns.color_palette("Paired", as_cmap=False)
cmap = sns.color_palette("pastel", as_cmap=False)
# cmap = np.array([[255, 105, 97], [97, 168, 255]]) / 255
# cmap = np.array([[124, 230, 199], [255, 169, 132]]) / 255
f, ax = plt.subplots(1, 1, figsize=(2.5, 5))
sns.despine(left=True, bottom=True)
ax.tick_params(pad=-2.5)

# nulls
sns.violinplot(data=df_prediction_null, ax=ax, x='B', y='score', hue='energy', split=True, scale='width', palette=cmap,
               inner=None, linewidth=0.5)
for violin in ax.collections:
    violin.set_alpha(0.2)

# observed
sns.violinplot(data=df_prediction, ax=ax, x='B', y='score', hue='energy', split=True, scale='width', palette=cmap,
               inner=None, linewidth=1.5)
n_violins = len(ax.collections)
for violin in ax.collections[int(n_violins/2):]:
    violin.set_alpha(1)

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles[:2], labels=['energy', 'neurobiology'], title='', bbox_to_anchor=(1, 1.15), loc='upper right')
ax.set_title('')
# ax.set_title(y_label)
ax.set_ylabel('neg[{0}] (higher = better)'.format(score.upper()))
ax.set_xlabel('')
ax.set_xticklabels(['unweighted', 'CT', 'CBF'])
ax.set_ylim([-0.93, -.86])

# add permutation test p-values
for i, B in enumerate(['wb', 'ct', 'cbf']):
    x = np.mean(df_prediction.loc[np.logical_and(df_prediction['B'] == B,
                                                 df_prediction['energy'] == 'yes'), 'score'])
    null = df_prediction_null.loc[np.logical_and(df_prediction_null['B'] == B,
                                                 df_prediction_null['energy'] == 'yes'), 'score']
    p_val = get_null_p(x, null)
    if p_val < 0.05:
        textstr = '*'
        ax.text(i - 0.1, x, textstr, rotation=90, fontweight='bold',
                horizontalalignment='center', verticalalignment='center')
    else:
        textstr = 'n.s.'
        ax.text(i - 0.1, x, textstr, rotation=90,
                horizontalalignment='center', verticalalignment='center')

    if B != 'wb':
        x = np.mean(df_prediction.loc[np.logical_and(df_prediction['B'] == B,
                                                     df_prediction['energy'] == 'no'), 'score'])
        null = df_prediction_null.loc[np.logical_and(df_prediction_null['B'] == B,
                                                     df_prediction_null['energy'] == 'no'), 'score']
        p_val = get_null_p(x, null)
        if p_val < 0.05:
            textstr = '*'
            ax.text(i + 0.1, x, textstr, rotation=270, fontweight='bold',
                    horizontalalignment='center', verticalalignment='center')
        else:
            textstr = 'n.s.'
            ax.text(i + 0.1, x, textstr, rotation=270,
                    horizontalalignment='center', verticalalignment='center')


f.savefig(os.path.join(environment.figdir, 'prediction_{0}.png'.format(y_name)), dpi=300, bbox_inches='tight',
          pad_inches=0.1)
plt.close()
