import os
from pfactor_gradients.pnc import Environment, Subject
from pfactor_gradients.routines import LoadSC, LoadCT, LoadRLFP, LoadCBF, LoadREHO, LoadALFF, LoadAverageBrainMaps
from pfactor_gradients.imaging_derivs import DataVector
from pfactor_gradients.pipelines import ComputeGradients
from pfactor_gradients.plotting import my_regplot

# %% Plotting
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='white', context='talk', font_scale=1)
import matplotlib.font_manager as font_manager
fontpath = '/Users/lindenmp/Library/Fonts/PublicSans-Thin.ttf'
prop = font_manager.FontProperties(fname=fontpath)
plt.rcParams['font.family'] = prop.get_name()
plt.rcParams['svg.fonttype'] = 'none'

# %% Setup project environment
computer = 'macbook'
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

grad = DataVector(data=compute_gradients.gradients[:, 0], name='grad_1')
# grad.rankdata()
# grad.rescale_unit_interval()
grad.brain_surface_plot(environment)

grad = DataVector(data=compute_gradients.gradients[:, 1], name='grad_2')
# grad.rankdata()
# grad.rescale_unit_interval()
grad.brain_surface_plot(environment)

# %% Load sc data
load_sc = LoadSC(environment=environment, Subject=Subject)
load_sc.run()
# refilter environment due to LoadSC excluding on disconnected nodes
environment.df = load_sc.df.copy()

# %% load mean brain maps
loaders_dict = {
    'ct': LoadCT(environment=environment, Subject=Subject),
    'rlfp': LoadRLFP(environment=environment, Subject=Subject),
    'cbf': LoadCBF(environment=environment, Subject=Subject),
    'reho': LoadREHO(environment=environment, Subject=Subject),
    'alff': LoadALFF(environment=environment, Subject=Subject)
}

load_average_bms = LoadAverageBrainMaps(loaders_dict=loaders_dict)
load_average_bms.run(return_descending=True)

# %% spatial correlations with grad
for key in load_average_bms.brain_maps:
    f, ax = plt.subplots(1, 2, figsize=(10, 5))
    my_regplot(compute_gradients.gradients[:, 0], load_average_bms.brain_maps[key].data,
               'Gradient 1', load_average_bms.brain_maps[key].name, ax[0])
    my_regplot(compute_gradients.gradients[:, 1], load_average_bms.brain_maps[key].data,
               'Gradient 2', load_average_bms.brain_maps[key].name, ax[1])
    f.savefig(os.path.join(environment.figdir, 'corr(grads,{0}).png'.format(load_average_bms.brain_maps[key].name)),
              dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close()
