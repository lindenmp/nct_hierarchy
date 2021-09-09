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

# %% help funcs
def helper_func(x_pos, y_pos, p_val, energy):
    if p_val < 0.05:
        # textstr = '$\mathit{p}_{FDR}$<.05'
        textstr = '*'
        fontweight = 'bold'
    else:
        textstr = 'n.s.'
        fontweight = 'normal'

    if energy == True:
        rotation = 270
        offset = 0.1
    else:
        rotation = 90
        offset = -0.1

    ax.text(x_pos + offset, y_pos, textstr, rotation=rotation, fontweight=fontweight, fontsize=8,
            horizontalalignment='center', verticalalignment='center')


def load_data(B, file_prefix, environment):
    # laod observed
    accuracy_mean = np.loadtxt(os.path.join(environment.pipelinedir, 'prediction',
                                            '{0}accuracy_mean.txt'.format(file_prefix)))
    df_pred = pd.DataFrame(columns=['score', 'B', 'energy'])
    df_pred['score'] = accuracy_mean
    df_pred['B'] = B
    if 'energy' in B:
        df_pred['energy'] = True
    else:
        df_pred['energy'] = False

    # load null
    accuracy_perm = np.loadtxt(os.path.join(environment.pipelinedir, 'prediction',
                                            '{0}accuracy_perm.txt'.format(file_prefix)))
    df_null = pd.DataFrame(columns=['score', 'B', 'energy'])
    df_null['score'] = accuracy_perm
    df_null['B'] = B
    if 'energy' in B:
        df_null['energy'] = True
    else:
        df_null['energy'] = False

    return df_pred, df_null

# %% load prediction results
y_name = 'F1_Exec_Comp_Res_Accuracy' # 'F1_Exec_Comp_Res_Accuracy' 'F1_Complex_Reasoning_Efficiency' 'F3_Executive_Efficiency'
y_label = 'Cognition'
c_name = 'asvm'
alg = 'rr'
score = 'rmse'
runpca = '80%'

B_list = ['energy-wb', 'ct', 'sa', 'energy-ct', 'energy-sa']
# B_list = ['energy-wb', 'ct', 'sa', 'energy-ct_flip', 'energy-sa_flip']
# B_list = ['energy-wb', 'ct', 'sa', 'energy-ct', 'energy-sa', 'energy-ct_flip', 'energy-sa_flip']

my_list = [0, 1, 2, 1, 2]
# my_list = [0, 1, 2, 1, 2, 1, 2]

df_pred = pd.DataFrame(columns=['score', 'B', 'energy'])
df_null = pd.DataFrame(columns=['score', 'B', 'energy'])

for B in B_list:
    # observed prediction performance
    file_prefix = 'histg2_{0}-{1}-{2}_alg-{3}_score-{4}_pca-{5}_'.format(B, y_name, c_name, alg, score, runpca)
    df_pred_tmp, df_null_tmp = load_data(B, file_prefix, environment)
    df_pred = pd.concat((df_pred, df_pred_tmp), axis=0)
    df_null = pd.concat((df_null, df_null_tmp), axis=0)

print(df_pred)

# %% compute and correct p-values for permutation test
p_vals_perm = np.zeros(len(B_list))
for i, B in enumerate(B_list):
    x = np.mean(df_pred.loc[df_pred['B'] == B, 'score'])
    null = df_null.loc[df_null['B'] == B, 'score']
    p_vals_perm[i] = get_null_p(x, null)

p_vals_perm = get_fdr_p(p_vals_perm)
print(p_vals_perm < 0.05)

# %% compute and correct p-values for pair wise comparisons
p_vals_dist = []

# 1) unweighted energy vs weight energy
x = df_pred.loc[df_pred['B'] == B_list[0], 'score']
y = df_pred.loc[df_pred['B'] == B_list[3], 'score']
# p_vals_dist.append(get_exact_p(x, y))
p_vals_dist.append(sp.stats.wilcoxon(x, y)[1])
y = df_pred.loc[df_pred['B'] == B_list[4], 'score']
# p_vals_dist.append(get_exact_p(x, y))
p_vals_dist.append(sp.stats.wilcoxon(x, y)[1])

# 2) weighted energy vs just neuro properties
x = df_pred.loc[df_pred['B'] == B_list[3], 'score']
y = df_pred.loc[df_pred['B'] == B_list[1], 'score']
# p_vals_dist.append(get_exact_p(x, y))
p_vals_dist.append(sp.stats.wilcoxon(x, y)[1])

x = df_pred.loc[df_pred['B'] == B_list[4], 'score']
y = df_pred.loc[df_pred['B'] == B_list[2], 'score']
# p_vals_dist.append(get_exact_p(x, y))
p_vals_dist.append(sp.stats.wilcoxon(x, y)[1])

p_vals_dist = np.array(p_vals_dist)
p_vals_dist = get_fdr_p(p_vals_dist)
print(p_vals_dist < 0.05)

# %% plot
df_pred['B_strip'] = df_pred['B'].map(lambda x: x.lstrip('energy-'))
df_null['B_strip'] = df_null['B'].map(lambda x: x.lstrip('energy-'))
# try:
#     df_pred['B_strip'] = df_pred['B_strip'].map(lambda x: x.rstrip('_flip'))
#     df_null['B_strip'] = df_null['B_strip'].map(lambda x: x.rstrip('_flip'))
# except:
#     pass

# cmap = sns.color_palette("Paired", as_cmap=False)
cmap = sns.color_palette("pastel", as_cmap=False)
# cmap = np.array([[255, 105, 97], [97, 168, 255]]) / 255
# cmap = np.array([[124, 230, 199], [255, 169, 132]]) / 255

f, ax = plt.subplots(1, 1, figsize=(2.5, 4))
sns.despine(left=True, bottom=True)
ax.tick_params(pad=-2.5)

# nulls (background)
sns.violinplot(data=df_null, ax=ax, x='B_strip', y='score', hue='energy', split=True, scale='width', palette=cmap,
               inner=None, linewidth=0.5)
for violin in ax.collections:
    violin.set_alpha(0.2)

# observed (foreground)
sns.violinplot(data=df_pred, ax=ax, x='B_strip', y='score', hue='energy', split=True, scale='width', palette=cmap,
               inner=None, linewidth=1.5)
n_violins = len(ax.collections)
for violin in ax.collections[int(n_violins/2):]:
    violin.set_alpha(1)

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles[:2], labels=['neurobiology only', 'energy'], title='', bbox_to_anchor=(1, 1.15), loc='upper right')
ax.set_ylabel('negative {0} (higher = better)'.format(score.upper()))
ax.set_xlabel('')
ax.set_xticklabels(['unweighted', 'CT', 'SA'])
if score == 'rmse':
    ax.set_ylim([-0.975, -0.895])

# add permutation p-values to plot
for i, B in enumerate(B_list):
    x_pos = my_list[i]
    y_pos = np.mean(df_pred.loc[df_pred['B'] == B, 'score'])
    if 'energy' in B:
        helper_func(x_pos, y_pos, p_vals_perm[i], True)
    else:
        helper_func(x_pos, y_pos, p_vals_perm[i], False)


f.savefig(os.path.join(environment.figdir, 'prediction_{0}'.format(y_name)), dpi=600, bbox_inches='tight',
          pad_inches=0.1)
plt.close()

# %%
B_list = ['energy-ct', 'energy-sa', 'energy-ct_flip', 'energy-sa_flip']

df_pred = pd.DataFrame(columns=['score', 'B', 'energy'])
df_null = pd.DataFrame(columns=['score', 'B', 'energy'])

for B in B_list:
    # observed prediction performance
    file_prefix = 'histg2_{0}-{1}-{2}_alg-{3}_score-{4}_pca-{5}_'.format(B, y_name, c_name, alg, score, runpca)
    df_pred_tmp, df_null_tmp = load_data(B, file_prefix, environment)
    df_pred = pd.concat((df_pred, df_pred_tmp), axis=0)
    df_null = pd.concat((df_null, df_null_tmp), axis=0)

# comparisons
x = df_pred.loc[df_pred['B'] == B_list[0], 'score']
y = df_pred.loc[df_pred['B'] == B_list[2], 'score']
print(get_exact_p(x, y))
print(sp.stats.wilcoxon(x, y))

x = df_pred.loc[df_pred['B'] == B_list[1], 'score']
y = df_pred.loc[df_pred['B'] == B_list[3], 'score']
print(get_exact_p(x, y))
print(sp.stats.wilcoxon(x, y))

df_pred['standard'] = ~df_pred['B'].str.contains('_flip')
df_pred['B_strip'] = df_pred['B'].map(lambda x: x.rstrip('_flip'))
df_null['standard'] = ~df_null['B'].str.contains('_flip')
df_null['B_strip'] = df_null['B'].map(lambda x: x.rstrip('_flip'))

f, ax = plt.subplots(1, 1, figsize=(2.5, 4))
sns.despine(left=True, bottom=True)
ax.tick_params(pad=-2.5)

# nulls (background)
sns.violinplot(data=df_null, ax=ax, x='B_strip', y='score', hue='standard', split=True, scale='width', palette=cmap,
               inner=None, linewidth=0.5)
for violin in ax.collections:
    violin.set_alpha(0.2)

# observed (foreground)
sns.violinplot(data=df_pred, ax=ax, x='B_strip', y='score', hue='standard', split=True, scale='width', palette=cmap,
               inner=None, linewidth=1.5)
n_violins = len(ax.collections)
for violin in ax.collections[int(n_violins/2):]:
    violin.set_alpha(1)

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles[:2], labels=['flipped', 'standard'], title='', bbox_to_anchor=(1, 1.15), loc='upper right')
ax.set_ylabel('negative {0} (higher = better)'.format(score.upper()))
ax.set_xlabel('')
ax.set_xticklabels(['CT', 'SA'])
# if score == 'rmse':
#     ax.set_ylim([-0.975, -0.895])

f.savefig(os.path.join(environment.figdir, 'prediction_flip_compare'), dpi=600, bbox_inches='tight',
          pad_inches=0.1)
plt.close()
