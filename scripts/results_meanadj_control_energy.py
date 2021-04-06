import numpy as np

from data_loader.pnc import Environment, Subject
from data_loader.routines import LoadSC, LoadCT
from utils.imaging_derivs import DataVector
from data_loader.pipelines import ComputeGradients, ComputeMinimumControlEnergy
from utils.utils import get_disc_repl

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
parc = 'schaefer'
n_parcels = 400
sc_edge_weight = 'streamlineCount'
environment = Environment(parc=parc, n_parcels=n_parcels, sc_edge_weight=sc_edge_weight)
environment.make_output_dirs()
environment.load_parc_data()

# %% get clustered gradients
filters = {'healthExcludev2': 0, 't1Exclude': 0,
           'b0ProtocolValidationStatus': 1, 'dti64ProtocolValidationStatus': 1, 'dti64Exclude': 0,
           'psychoactiveMedPsychv2': 0, 'restProtocolValidationStatus': 1, 'restExclude': 0}
environment.load_metadata(filters)
grad_pipeline = ComputeGradients(environment=environment, Subject=Subject)
grad_pipeline.run()

# %% get mean structural A matrix
filters = {'healthExcludev2': 0, 't1Exclude': 0,
           'b0ProtocolValidationStatus': 1, 'dti64ProtocolValidationStatus': 1, 'dti64Exclude': 0,
           'psychoactiveMedPsychv2': 0}
environment.load_metadata(filters)

# retain half as discovery set
environment.df['disc_repl'] = get_disc_repl(environment.df, frac=0.5)
environment.df = environment.df.loc[environment.df['disc_repl'] == 0, :]
print(environment.df.shape)

# Load sc data
sc_pipeline = LoadSC(environment=environment, Subject=Subject)
sc_pipeline.run()

# %% get brain map for biasing energy

# Load ct data
ct_pipeline = LoadCT(environment=environment, Subject=Subject)
ct_pipeline.run()
ct = DataVector(data=np.nanmean(ct_pipeline.ct, axis=0), name='ct')
ct.rescale_unit_interval()
ct.brain_surface_plot(environment)

# %%
nct_pipeline = ComputeMinimumControlEnergy(environment=environment,
                                           grad_pipeline=grad_pipeline, sc_pipeline=sc_pipeline,
                                           spars_thresh=0.06, control='minimum', T=1, B='x0xfwb')
nct_pipeline.run_average_adj()

# %%
nct_pipeline = ComputeMinimumControlEnergy(environment=environment,
                                           grad_pipeline=grad_pipeline, sc_pipeline=sc_pipeline,
                                           spars_thresh=0.06, control='minimum', T=1, B='wb')
nct_pipeline.run_average_adj()

# %%
nct_pipeline = ComputeMinimumControlEnergy(environment=environment,
                                           grad_pipeline=grad_pipeline, sc_pipeline=sc_pipeline,
                                           spars_thresh=0.06, control='minimum', T=1, B=ct)
nct_pipeline.run_average_adj()
