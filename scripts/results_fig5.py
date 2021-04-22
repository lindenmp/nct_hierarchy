# %%
import sys, os, platform
import numpy as np
import pandas as pd

if platform.system() == 'Linux':
    sys.path.extend(['/cbica/home/parkesl/research_projects/pfactor_gradients'])
from pfactor_gradients.pnc import Environment, Subject

# %% Setup project environment
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='white', context='paper', font_scale=1)
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
# environment.load_parc_data()

# %% load prediction results
B_list = ['ct', 'cbf', 'reho', 'alff', 'energy-ct', 'energy-cbf', 'energy-reho', 'energy-alff']
n_B = len(B_list)
n_rand_splits = 100
p_vals = np.zeros(n_B)

y_name = 'Overall_Psychopathology' # 'Overall_Psychopathology' 'F3_Executive_Efficiency' 'F1_Exec_Comp_Res_Accuracy'
c_name = 'asvm'
alg = 'rr'
score = 'rmse'
runpca = '80%'

df_energy = pd.DataFrame(columns=['score', 'B', 'energy'])
file_prefix = '{0}-{1}-{2}_alg-{3}_score-{4}_pca-{5}_'.format('energy-wb', y_name, c_name,
                                                              alg, score, runpca)
accuracy_mean = np.loadtxt(os.path.join(environment.pipelinedir, 'prediction',
                                        '{0}accuracy_mean.txt'.format(file_prefix)))
df_tmp = pd.DataFrame(columns=['score', 'B', 'energy'])
df_tmp['score'] = accuracy_mean
df_tmp['B'] = 'wb'
df_tmp['energy'] = 'yes'
df_energy = pd.concat((df_energy, df_tmp), axis=0)

df_energy_null = pd.DataFrame(columns=['score', 'B', 'energy'])
accuracy_perm = np.loadtxt(os.path.join(environment.pipelinedir, 'prediction',
                                        '{0}accuracy_perm.txt'.format(file_prefix)))
df_tmp = pd.DataFrame(columns=['score', 'B', 'energy'])
df_tmp['score'] = accuracy_perm
df_tmp['B'] = 'wb'
df_tmp['energy'] = 'yes'
df_energy_null = pd.concat((df_energy_null, df_tmp), axis=0)

for i in np.arange(n_B):
    file_prefix = '{0}-{1}-{2}_alg-{3}_score-{4}_pca-{5}_'.format(B_list[i], y_name, c_name,
                                                                  alg, score, runpca)
    accuracy_mean = np.loadtxt(os.path.join(environment.pipelinedir, 'prediction',
                                            '{0}accuracy_mean.txt'.format(file_prefix)))
    df_tmp = pd.DataFrame(columns=['score', 'B', 'energy'])
    df_tmp['score'] = accuracy_mean
    if len(B_list[i].split('-')) == 2:
        df_tmp['B'] = B_list[i].split('-')[1]
        df_tmp['energy'] = 'yes'
    elif len(B_list[i].split('-')) == 1:
        df_tmp['B'] = B_list[i].split('-')[0]
        df_tmp['energy'] = 'no'
    df_energy = pd.concat((df_energy, df_tmp), axis=0)

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
    df_energy_null = pd.concat((df_energy_null, df_tmp), axis=0)

# %%
# cmap = sns.color_palette("Paired", as_cmap=False)
# cmap = np.array([[255, 105, 97], [97, 168, 255]]) / 255
cmap = np.array([[124, 230, 199], [255, 169, 132]]) / 255
f, ax = plt.subplots(1, 1, figsize=(5, 2.5))

# nulls
sns.violinplot(data=df_energy_null, ax=ax, x='B', y='score', hue='energy', split=True, scale='width', palette=cmap,
               inner=None, linewidth=1)
for violin in ax.collections:
    violin.set_alpha(0.2)

# observed
sns.violinplot(data=df_energy, ax=ax, x='B', y='score', hue='energy', split=True, scale='width', palette=cmap,
               inner='quartiles', linewidth=1.5)
n_violins = len(ax.collections)
for violin in ax.collections[int(n_violins/2):]:
    violin.set_alpha(1)

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles[:2], labels=labels[:2], title="NCT")
ax.set_title(y_name)
ax.set_ylabel(score)

ax.tick_params(pad=-2.5)
f.savefig(os.path.join(environment.figdir, 'fig-5_prediction_{0}.png'.format(y_name)), dpi=150, bbox_inches='tight',
          pad_inches=0.1)
plt.show()
plt.close()
