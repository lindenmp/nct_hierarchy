import os

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import seaborn as sns
import pandas as pd

from data_loader.routines import LoadSC, LoadRLFP, LoadCT
from utils.imaging_derivs import DataMatrix, DataVector
from utils.plotting import my_regplot
from utils.utils import get_pdist_clusters, get_disc_repl, mean_over_clusters

# %% Set general plotting params
sns.set(style='white', context='talk', font_scale=1)
import matplotlib.font_manager as font_manager

fontpath = '/Users/lindenmp/Library/Fonts/PublicSans-Thin.ttf'
prop = font_manager.FontProperties(fname=fontpath)
plt.rcParams['font.family'] = prop.get_name()
plt.rcParams['svg.fonttype'] = 'none'

# %% Setup project environment
from data_loader.pnc import Environment, Subject

parc = 'schaefer'
n_parcels = 400
sc_edge_weight = 'streamlineCount'
environment = Environment(parc=parc, n_parcels=n_parcels, sc_edge_weight=sc_edge_weight)
environment.make_output_dirs()
environment.load_parc_data()

# %% get clustered gradients
from data_loader.pipelines import ComputeGradients

filters = {'healthExcludev2': 0, 't1Exclude': 0,
           'b0ProtocolValidationStatus': 1, 'dti64ProtocolValidationStatus': 1, 'dti64Exclude': 0,
           'psychoactiveMedPsychv2': 0, 'restProtocolValidationStatus': 1, 'restExclude': 0}
environment.load_metadata(filters)
cg = ComputeGradients(environment=environment, Subject=Subject)
cg.run()

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
pipeline = LoadSC(environment=environment, Subject=Subject)
pipeline.run()
A = pipeline.A
n_subs = pipeline.df.shape[0]

# Get streamline count and network density
A_d = np.zeros((n_subs,))
for i in range(n_subs):
    A_d[i] = np.count_nonzero(np.triu(A[:, :, i])) / ((A[:, :, i].shape[0] ** 2 - A[:, :, i].shape[0]) / 2)

# Get group average adj. matrix
mean_spars = np.round(A_d.mean(), 2)
print(mean_spars)

A = np.mean(A, 2)
thresh = np.percentile(A, 100 - (mean_spars * 100))
A[A < thresh] = 0
print(np.count_nonzero(np.triu(A)) / ((A.shape[0] ** 2 - A.shape[0]) / 2))
A = DataMatrix(data=A)
A.get_distance_matrix()

# %% get bold rlfp
filters = {'healthExcludev2': 0, 't1Exclude': 0,
           'b0ProtocolValidationStatus': 1, 'dti64ProtocolValidationStatus': 1, 'dti64Exclude': 0,
           'psychoactiveMedPsychv2': 0, 'restProtocolValidationStatus': 1, 'restExclude': 0}
environment.load_metadata(filters)

# retain half as discovery set
environment.df['disc_repl'] = get_disc_repl(environment.df, frac=0.5)
environment.df = environment.df.loc[environment.df['disc_repl'] == 0, :]
print(environment.df.shape)

# Load rlfp data
pipeline = LoadRLFP(environment=environment, Subject=Subject)
pipeline.run()
rlfp = pipeline.rlfp

# mean over subjects
rlfp_mean = DataVector(data=np.mean(rlfp, axis=0))
rlfp_mean.regress_nuisance(c=A.hops.mean(axis=0))

# %% get cortical thickness
filters = {'healthExcludev2': 0, 't1Exclude': 0,
           'b0ProtocolValidationStatus': 1, 'dti64ProtocolValidationStatus': 1, 'dti64Exclude': 0,
           'psychoactiveMedPsychv2': 0}
environment.load_metadata(filters)

# retain half as discovery set
environment.df['disc_repl'] = get_disc_repl(environment.df, frac=0.5)
environment.df = environment.df.loc[environment.df['disc_repl'] == 0, :]
print(environment.df.shape)

# Load ct data
pipeline = LoadCT(environment=environment, Subject=Subject)
pipeline.run()
ct = pipeline.ct

# mean over subjects
ct_mean = DataVector(data=np.mean(ct, axis=0))
ct_mean.regress_nuisance(c=A.hops.mean(axis=0))

# %% get gene expression
from data_loader.pipelines import LoadGeneExpression
pipeline = LoadGeneExpression(environment=environment)
pipeline.run()
expression = pipeline.expression

# SST/PVALB expression
sst_z = sp.stats.zscore(expression.loc[:, 'SST'], nan_policy='omit')
pvalb_z = sp.stats.zscore(expression.loc[:, 'PVALB'], nan_policy='omit')
sst_pvalb_delta = sst_z - pvalb_z
sst_pvalb_delta[np.isnan(sst_pvalb_delta)] = np.nanmedian(sst_pvalb_delta)
sst_pvalb_delta = DataVector(data=sst_pvalb_delta)
sst_pvalb_delta.regress_nuisance(c=A.hops.mean(axis=0))

# E:I
excitation = expression.loc[:, ['GRIA1', 'GRIA2', 'GRIA3', 'GRIA4', 'GRIN1', 'GRIN2A', 'GRIN2B', 'GRIN2C']].sum(axis=1)
inhibition = expression.loc[:, ['GABRA1', 'GABRA2', 'GABRA3', 'GABRA4', 'GABRA5', 'GABRB1',
                                'GABRB2', 'GABRB3', 'GABRG1', 'GABRG2', 'GABRG3']].sum(axis=1)
# inhibition = expression.loc[:, ['SST', 'PVALB', 'GABRA1', 'GABRA2', 'GABRA3', 'GABRA4', 'GABRA5', 'GABRB1',
#                                 'GABRB2', 'GABRB3', 'GABRG1', 'GABRG2', 'GABRG3']].sum(axis=1)
# inhibition = expression.loc[:, ['SST', 'PVALB']].sum(axis=1)
ei_ratio = excitation.values/inhibition.values
ei_ratio[np.isnan(ei_ratio)] = np.nanmedian(ei_ratio)
ei_ratio = DataVector(data=ei_ratio)
ei_ratio.regress_nuisance(c=A.hops.mean(axis=0))

# %% Plots
print('\nGenerating figures')
gradient = cg.gradients[:, 0]

def my_gradient_plot(A, environment, figname='figure.png', plot_resid=False):
    grad_slope = DataMatrix(data=A.grad_slope)
    grad_slope.regress_nuisance(c=A.hops)
    grad_r2 = DataMatrix(data=A.grad_r2)
    grad_r2.regress_nuisance(c=A.hops)
    grad_resid = DataMatrix(data=A.grad_resid)
    grad_resid.regress_nuisance(c=A.hops)
    grad_var = DataMatrix(data=A.grad_var)
    grad_var.regress_nuisance(c=A.hops)

    if plot_resid:
        to_plot = [grad_slope.data_resid, grad_r2.data_resid, grad_resid.data_resid, grad_var.data_resid]
    else:
        to_plot = [grad_slope.data, grad_r2.data, grad_resid.data, grad_var.data]
    to_plot_labels = ['Slope', 'R2', 'RMSE', 'Variance']
    to_plot_ylims = [[-1, 1], [-0.5, 0.5], [-0.3, 0.5], [-0.5, 1]]
    n_subplots = len(to_plot)

    f, ax = plt.subplots(3, n_subplots, figsize=(n_subplots * 5, 15))
    for i in np.arange(n_subplots):
        indices = np.where(~np.eye(n_parcels, dtype=bool) * ~np.isnan(to_plot[i]))
        stats = sp.stats.pearsonr(A.hops[indices], to_plot[i][indices])
        print('Correlation between hops and {:}: {:.2f}, {:.2f}'.format(to_plot_labels[i], stats[0], stats[1]))

        my_regplot(A.gradient, np.nanmean(np.where(to_plot[i] != 0, to_plot[i], np.nan), axis=1),
                   'Gradient', to_plot_labels[i] + ' (mean)', ax[0, i])
        # ax[0, i].set_ylim(to_plot_ylims[i])

        my_regplot(A.gradient, np.nanstd(np.where(to_plot[i] != 0, to_plot[i], np.nan), axis=1),
                   'Gradient', to_plot_labels[i] + ' (std)', ax[1, i])
        # ax[1, i].set_ylim([0, 1])

        my_regplot(np.nanmean(A.hops, axis=1), np.nanmean(np.where(to_plot[i] != 0, to_plot[i], np.nan), axis=1),
                   'Hops (mean)', to_plot_labels[i] + ' (mean)', ax[2, i])
        # ax[1, i].set_ylim([0, 1])

    f.subplots_adjust(wspace=0.5)
    f.savefig(os.path.join(environment.figdir, figname),
              dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close()

# %% 1) paths
A.get_gradient_slopes(gradient, method='linear')
my_gradient_plot(A, environment, figname='path_traversal_gradient.png', plot_resid=False)
my_gradient_plot(A, environment, figname='path_traversal_gradient_resid.png', plot_resid=True)

# %% 2) rlfp
A.get_gradient_slopes(gradient, x_map=rlfp_mean.data_resid, method='linear')
my_gradient_plot(A, environment, figname='path_traversal_rlfp.png')

# %% 3) ct
A.get_gradient_slopes(gradient, x_map=ct_mean.data_resid, method='linear')
my_gradient_plot(A, environment, figname='path_traversal_ct.png')

# %% 4) sst - pvalb
A.get_gradient_slopes(gradient, x_map=sst_pvalb_delta.data_resid, method='linear')
my_gradient_plot(A, environment, figname='path_traversal_sst-pvalb.png')

# %% 5) E:I
A.get_gradient_slopes(gradient, x_map=ei_ratio.data_resid, method='linear')
my_gradient_plot(A, environment, figname='path_traversal_ei.png')
