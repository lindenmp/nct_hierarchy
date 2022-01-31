# %% import
import sys, os, platform
from pfactor_gradients.pipelines import DCM

# %% import workspace
os.environ["MY_PYTHON_WORKSPACE"] = 'ave_adj'
os.environ["WHICH_BRAIN_MAP"] = 'hist-g2'
# os.environ["WHICH_BRAIN_MAP"] = 'func-g1'
from setup_workspace import *

# %% plotting
import seaborn as sns
import matplotlib.pyplot as plt
from pfactor_gradients.plotting import set_plotting_params
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

# %%
# rsts = dcm.rsts_states
# print(rsts.shape)

# spmdir = '/Users/lindenmp/Google-Drive-Penn/work/matlab_tools/spm12'
# outdir = '/Users/lindenmp/Google-Drive-Penn/work/research_projects/pfactor_gradients/output_cluster/pnc/schaefer_400_streamlineCount/pipelines/spdcm'

# octave.eval("rand('state',%i)" % 1)
# octave.spdcm_firstlevel_loop(spmdir, rsts, environment.rsfmri_tr, environment.rsfmri_te, outdir)
