# %% import
import sys, os, platform
from pfactor_gradients.imaging_derivs import DataMatrix

import scipy as sp

# %% import workspace
os.environ["MY_PYTHON_WORKSPACE"] = 'ave_adj'
os.environ["WHICH_BRAIN_MAP"] = 'hist-g2'
# os.environ["WHICH_BRAIN_MAP"] = 'func-g1'
from setup_workspace import *

# %% plotting
import seaborn as sns
import matplotlib.pyplot as plt
from pfactor_gradients.plotting import set_plotting_params
set_plotting_params(format='svg')
figsize = 1.5

# %% brain maps
A_tmp = DataMatrix(data=A)
A_tmp.get_strength()

for key in brain_maps:
    nan_mask = np.isnan(brain_maps[key].data)

    print('state_brain_map vs. {0}'.format(key),
          sp.stats.pearsonr(state_brain_map[~nan_mask], brain_maps[key].data[~nan_mask]))
    print('strength vs. {0}'.format(key),
          sp.stats.pearsonr(A_tmp.S[~nan_mask], brain_maps[key].data[~nan_mask]))

    # plot brain map
    brain_maps[key].brain_surface_plot(environment)

print('strength vs. state_brain_map', sp.stats.pearsonr(A_tmp.S, state_brain_map))

# plot state brain map
DataVector(data=state_brain_map, name='state_brain_map').brain_surface_plot(environment)
DataVector(data=states == 0, name='state_0').brain_surface_plot(environment)
DataVector(data=states == int(n_states/2), name='state_{0}'.format(int(n_states/2))).brain_surface_plot(environment)
DataVector(data=states == n_states-1, name='state_{0}'.format(int(n_states-1))).brain_surface_plot(environment)

