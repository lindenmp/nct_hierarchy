# %%
import sre_parse
import sys, os, platform
from pfactor_gradients.pnc import Environment
from pfactor_gradients.utils import get_exact_p, get_null_p, get_fdr_p, get_p_val_string
import numpy as np
import pandas as pd
import scipy as sp

# %% Plotting
import seaborn as sns
import matplotlib.pyplot as plt
from pfactor_gradients.plotting import set_plotting_params
set_plotting_params(format='png')

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
df_pred = pd.DataFrame(columns=['score', 'B', 'energy'])
file_prefix = '{0}-{1}-{2}_alg-{3}_score-{4}_pca-{5}_'.format('energy-wb', y_name, c_name, alg, score, runpca)
accuracy_mean = np.loadtxt(os.path.join(environment.pipelinedir, 'prediction',
                                        '{0}accuracy_mean.txt'.format(file_prefix)))
df_tmp = pd.DataFrame(columns=['score', 'B', 'energy'])
df_tmp['score'] = accuracy_mean
df_tmp['B'] = 'wb'
df_tmp['energy'] = 'yes'
df_pred = pd.concat((df_pred, df_tmp), axis=0)

# load unweighted energy null
df_pred_null = pd.DataFrame(columns=['score', 'B', 'energy'])
accuracy_perm = np.loadtxt(os.path.join(environment.pipelinedir, 'prediction', '{0}accuracy_perm.txt'.format(file_prefix)))
df_tmp = pd.DataFrame(columns=['score', 'B', 'energy'])
df_tmp['score'] = accuracy_perm
df_tmp['B'] = 'wb'
df_tmp['energy'] = 'yes'
df_pred_null = pd.concat((df_pred_null, df_tmp), axis=0)

# load prediction from brain maps and weighted energgy
B_list = ['ct', 'cbf', 'energy-ct', 'energy-cbf']
# B_list = ['ct', 'cbf', 'energy-ct-u', 'energy-cbf-u']
# B_list = ['ct', 'cbf', 'energy-ct-l', 'energy-cbf-l']
n_B = len(B_list)
p_vals = np.zeros(n_B)

for i in np.arange(n_B):
    # observed prediction performance
    file_prefix = '{0}-{1}-{2}_alg-{3}_score-{4}_pca-{5}_'.format(B_list[i], y_name, c_name, alg, score, runpca)
    accuracy_mean = np.loadtxt(os.path.join(environment.pipelinedir, 'prediction', '{0}accuracy_mean.txt'.format(file_prefix)))
    df_tmp = pd.DataFrame(columns=['score', 'B', 'energy'])
    df_tmp['score'] = accuracy_mean
    if len(B_list[i].split('-')) >= 2:
        df_tmp['B'] = B_list[i].split('-')[1]
        df_tmp['energy'] = 'yes'
    elif len(B_list[i].split('-')) == 1:
        df_tmp['B'] = B_list[i].split('-')[0]
        df_tmp['energy'] = 'no'
    df_pred = pd.concat((df_pred, df_tmp), axis=0)

    # null prediction performance
    accuracy_perm = np.loadtxt(os.path.join(environment.pipelinedir, 'prediction', '{0}accuracy_perm.txt'.format(file_prefix)))
    df_tmp = pd.DataFrame(columns=['score', 'B', 'energy'])
    df_tmp['score'] = accuracy_perm
    if len(B_list[i].split('-')) == 2:
        df_tmp['B'] = B_list[i].split('-')[1]
        df_tmp['energy'] = 'yes'
    elif len(B_list[i].split('-')) == 1:
        df_tmp['B'] = B_list[i].split('-')[0]
        df_tmp['energy'] = 'no'
    df_pred_null = pd.concat((df_pred_null, df_tmp), axis=0)

# %% compute and correct p-values for permutation test
my_list = [['wb', 'yes', 0], ['ct', 'yes', 1], ['ct', 'no', 1], ['cbf', 'yes', 2], ['cbf', 'no', 2]]
p_vals_perm = np.zeros(len(my_list))
for i in np.arange(len(my_list)):
    B = my_list[i][0]
    energy = my_list[i][1]
    x = np.mean(df_pred.loc[np.logical_and(df_pred['B'] == B, df_pred['energy'] == energy), 'score'])
    null = df_pred_null.loc[np.logical_and(df_pred_null['B'] == B, df_pred_null['energy'] == energy), 'score']
    p_vals_perm[i] = get_null_p(x, null)

p_vals_perm = get_fdr_p(p_vals_perm)
print(p_vals_perm < 0.05)

# %% compute and correct p-values for pair wise comparisons
p_vals_dist = []

# 1) unweighted energy vs weight energy
x = df_pred.loc[np.logical_and(df_pred['B'] == 'wb', df_pred['energy'] == 'yes'), 'score']
y = df_pred.loc[np.logical_and(df_pred['B'] == 'ct', df_pred['energy'] == 'yes'), 'score']
# p_vals_dist.append(get_exact_p(x, y))
p_vals_dist.append(sp.stats.wilcoxon(x, y)[1])
y = df_pred.loc[np.logical_and(df_pred['B'] == 'cbf', df_pred['energy'] == 'yes'), 'score']
# p_vals_dist.append(get_exact_p(x, y))
p_vals_dist.append(sp.stats.wilcoxon(x, y)[1])

# 2) weighted energy vs just neuro properties
x = df_pred.loc[np.logical_and(df_pred['B'] == 'ct', df_pred['energy'] == 'yes'), 'score']
y = df_pred.loc[np.logical_and(df_pred['B'] == 'ct', df_pred['energy'] == 'no'), 'score']
# p_vals_dist.append(get_exact_p(x, y))
p_vals_dist.append(sp.stats.wilcoxon(x, y)[1])

x = df_pred.loc[np.logical_and(df_pred['B'] == 'cbf', df_pred['energy'] == 'yes'), 'score']
y = df_pred.loc[np.logical_and(df_pred['B'] == 'cbf', df_pred['energy'] == 'no'), 'score']
# p_vals_dist.append(get_exact_p(x, y))
p_vals_dist.append(sp.stats.wilcoxon(x, y)[1])

p_vals_dist = np.array(p_vals_dist)
p_vals_dist = get_fdr_p(p_vals_dist)
print(p_vals_dist < 0.05)

# %% help funcs
def helper_func(x_pos, y_pos, p_val, energy):
    if p_val < 0.05:
        # textstr = '$\mathit{p}_{FDR}$<.05'
        textstr = '*'
        fontweight = 'bold'
    else:
        textstr = 'n.s.'
        fontweight = 'normal'

    if energy == 'yes':
        rotation = 90
        offset = -0.1
    else:
        rotation = 270
        offset = 0.1

    ax.text(x_pos + offset, y_pos, textstr, rotation=rotation, fontweight=fontweight, fontsize=8,
            horizontalalignment='center', verticalalignment='center')

# plot
# cmap = sns.color_palette("Paired", as_cmap=False)
cmap = sns.color_palette("pastel", as_cmap=False)
# cmap = np.array([[255, 105, 97], [97, 168, 255]]) / 255
# cmap = np.array([[124, 230, 199], [255, 169, 132]]) / 255

f, ax = plt.subplots(1, 1, figsize=(2.5, 4))
sns.despine(left=True, bottom=True)
ax.tick_params(pad=-2.5)

# nulls (background)
sns.violinplot(data=df_pred_null, ax=ax, x='B', y='score', hue='energy', split=True, scale='width', palette=cmap,
               inner=None, linewidth=0.5)
for violin in ax.collections:
    violin.set_alpha(0.2)

# observed (foreground)
sns.violinplot(data=df_pred, ax=ax, x='B', y='score', hue='energy', split=True, scale='width', palette=cmap,
               inner=None, linewidth=1.5)
n_violins = len(ax.collections)
for violin in ax.collections[int(n_violins/2):]:
    violin.set_alpha(1)

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles[:2], labels=['energy', 'neurobiology only'], title='', bbox_to_anchor=(1, 1.15), loc='upper right')
ax.set_ylabel('negative {0} (higher = better)'.format(score.upper()))
ax.set_xlabel('')
ax.set_xticklabels(['unweighted', 'CT', 'CBF'])
ax.set_ylim([-0.93, -.86])

# add permutation p-values to plot
for i in np.arange(len(my_list)):
    B = my_list[i][0]
    energy = my_list[i][1]
    x_pos = my_list[i][2]
    y_pos = np.mean(df_pred.loc[np.logical_and(df_pred['B'] == B, df_pred['energy'] == energy), 'score'])
    helper_func(x_pos, y_pos, p_vals_perm[i], energy)

f.savefig(os.path.join(environment.figdir, 'prediction_{0}'.format(y_name)), dpi=300, bbox_inches='tight',
          pad_inches=0.1)
plt.close()
