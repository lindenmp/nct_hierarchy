# %% import
import sys, os, platform
from src.pipelines import DCM

# %% import workspace
os.environ["MY_PYTHON_WORKSPACE"] = 'ave_adj'
os.environ["WHICH_BRAIN_MAP"] = 'hist-g2'
from setup_workspace import *

# %% plotting
import seaborn as sns
import matplotlib.pyplot as plt
from src.plotting import set_plotting_params
set_plotting_params(format='png')
figsize = 1.5

# %% DCM
dcm = DCM(environment=environment, Subject=Subject, states=states, file_prefix=which_brain_map, force_rerun=True)
# dcm.run_mean_ts()
# dcm.run_concat_ts()
dcm.run_concat_mean_ts()

# %%
f, ax = plt.subplots(1, 1, figsize=(15, 4))
sns.heatmap(dcm.rsts_states.transpose(), cmap='gray', ax=ax, center=0)
ax.tick_params(pad=-2.5)
f.savefig(os.path.join(environment.figdir, 'spdcm_ts_{0}.png'.format(which_brain_map)), dpi=500, bbox_inches='tight',
          pad_inches=0.1)
plt.close()
