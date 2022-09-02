# %% import
import sys, os, platform
from src.plotting import my_null_plot
from src.utils import get_null_p

from brainsmash.mapgen.base import Base
from tqdm import tqdm

# %% import workspace
os.environ["MY_PYTHON_WORKSPACE"] = 'ave_adj'
os.environ["WHICH_BRAIN_MAP"] = 'hist-g2'
from setup_workspace import *

# %% plotting
import seaborn as sns
import matplotlib.pyplot as plt
from src.plotting import set_plotting_params
set_plotting_params(format='svg')
figsize = 1.5

dv = DataVector(data=state_brain_map, name=which_brain_map)
dv.rankdata()
dv.brain_surface_plot(environment)

dv = DataVector(data=brain_map_loader.micro, name='micro')
dv.rankdata()
dv.brain_surface_plot(environment)

dv = DataVector(data=compute_gradients.gradients[:, 0], name='func')
dv.rankdata()
dv.brain_surface_plot(environment)

# %% generate surrogates using brainsmash
n_surrogates = 10000
file = 'brainsmash_surrogates_{0}_n{1}.npy'.format(which_brain_map, n_surrogates)
if os.path.exists(os.path.join(environment.pipelinedir, file)) == False:
    D = sp.spatial.distance.pdist(environment.centroids, 'euclidean')
    D = sp.spatial.distance.squareform(D)

    base = Base(x=state_brain_map, D=D, resample=True)
    surrogates = base(n=n_surrogates)

    np.save(os.path.join(environment.pipelinedir, file), surrogates)
else:
    surrogates = np.load(os.path.join(environment.pipelinedir, file))


# %% correlation(cyto,micro)
observed, _ = sp.stats.pearsonr(state_brain_map, brain_map_loader.micro)
null = np.zeros(n_surrogates)

for i in tqdm(np.arange(n_surrogates)):
    null[i], _ = sp.stats.pearsonr(surrogates[i, :], brain_map_loader.micro)

p_val = get_null_p(observed, null, version='standard', abs=True)
print(p_val)

f, ax = plt.subplots(1, 1, figsize=(figsize, figsize))
my_null_plot(observed=observed, null=null, p_val=p_val, xlabel=' ', ax=ax)
f.savefig(os.path.join(environment.figdir, 'spat_corr(cyto,micro).png'), dpi=600,
          bbox_inches='tight', pad_inches=0.01)
plt.close()

# %% correlation(cyto,func)
observed, _ = sp.stats.pearsonr(state_brain_map, compute_gradients.gradients[:, 0])
null = np.zeros(n_surrogates)

for i in tqdm(np.arange(n_surrogates)):
    null[i], _ = sp.stats.pearsonr(surrogates[i, :], compute_gradients.gradients[:, 0])

p_val = get_null_p(observed, null, version='standard', abs=True)
print(p_val)

f, ax = plt.subplots(1, 1, figsize=(figsize, figsize))
my_null_plot(observed=observed, null=null, p_val=p_val, xlabel=' ', ax=ax)
f.savefig(os.path.join(environment.figdir, 'spat_corr(cyto,func).png'), dpi=600,
          bbox_inches='tight', pad_inches=0.01)
plt.close()

# %% correlation(micro,func)
print(sp.stats.pearsonr(brain_map_loader.micro, compute_gradients.gradients[:, 0]))
