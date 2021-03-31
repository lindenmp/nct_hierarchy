import os

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import seaborn as sns
from sklearn.cluster import KMeans

from data_loader.routines import LoadSC, LoadRLFP, LoadCT
from utils.imaging_derivs import DataMatrix, DataVector
from utils.plotting import my_regplot, my_nullplot
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

# %% Plots
gradient = cg.gradients[:, 0]

# %% 1) paths
A.get_gradient_slopes(gradient)
indices = np.where(~np.eye(n_parcels, dtype=bool) * ~np.isnan(A.grad_resid))

grad_slope = DataMatrix(data=A.grad_slope)
grad_slope.regress_nuisance(c=A.hops)
grad_resid = DataMatrix(data=A.grad_resid)
grad_resid.regress_nuisance(c=A.hops)

stats = sp.stats.pearsonr(A.grad_slope[indices], A.hops[indices])
print('Correlation between slope and hops: {:.2f}, {:.2f}'.format(stats[0], stats[1]))
stats = sp.stats.pearsonr(grad_slope.data_resid[indices], A.hops[indices])
print('\t after regressing out hops: {:.2f}, {:.2f}'.format(stats[0], stats[1]))

stats = sp.stats.pearsonr(A.grad_resid[indices], A.hops[indices])
print('Correlation between error and hops: {:.2f}, {:.2f}'.format(stats[0], stats[1]))
stats = sp.stats.pearsonr(grad_resid.data_resid[indices], A.hops[indices])
print('\t after regressing out hops: {:.2f}, {:.2f}'.format(stats[0], stats[1]))

f, ax = plt.subplots(3, 1, figsize=(5, 5*3))
my_regplot(gradient, np.nanmean(np.where(grad_slope.data_resid != 0, grad_slope.data_resid, np.nan), axis=1),
           'Gradient', 'Slope (mean)', ax[0])
my_regplot(gradient, np.nanstd(np.where(grad_slope.data_resid != 0, grad_slope.data_resid, np.nan), axis=1),
           'Gradient', 'Slope (std)', ax[1])
my_regplot(gradient, np.nanmean(np.where(grad_resid.data_resid != 0, grad_resid.data_resid, np.nan), axis=1),
           'Gradient', 'RMSE (mean)', ax[2])
f.subplots_adjust(wspace=0.5)
f.savefig(os.path.join(environment.figdir, 'path_traversal_gradient.png'),
          dpi=150, bbox_inches='tight', pad_inches=0.1)
plt.close()

# %% 2) rlfp
A.get_gradient_slopes(gradient, x_map=rlfp_mean.data_resid)
indices = np.where(~np.eye(n_parcels, dtype=bool) * ~np.isnan(A.grad_resid))

grad_slope = DataMatrix(data=A.grad_slope)
grad_slope.regress_nuisance(c=A.hops)
grad_resid = DataMatrix(data=A.grad_resid)
grad_resid.regress_nuisance(c=A.hops)

stats = sp.stats.pearsonr(A.grad_slope[indices], A.hops[indices])
print('Correlation between slope and hops: {:.2f}, {:.2f}'.format(stats[0], stats[1]))
stats = sp.stats.pearsonr(grad_slope.data_resid[indices], A.hops[indices])
print('\t after regressing out hops: {:.2f}, {:.2f}'.format(stats[0], stats[1]))

stats = sp.stats.pearsonr(A.grad_resid[indices], A.hops[indices])
print('Correlation between error and hops: {:.2f}, {:.2f}'.format(stats[0], stats[1]))
stats = sp.stats.pearsonr(grad_resid.data_resid[indices], A.hops[indices])
print('\t after regressing out hops: {:.2f}, {:.2f}'.format(stats[0], stats[1]))

f, ax = plt.subplots(3, 1, figsize=(5, 5*3))
my_regplot(gradient, np.nanmean(np.where(grad_slope.data_resid != 0, grad_slope.data_resid, np.nan), axis=1),
           'Gradient', 'Slope (mean)', ax[0])
my_regplot(gradient, np.nanstd(np.where(grad_slope.data_resid != 0, grad_slope.data_resid, np.nan), axis=1),
           'Gradient', 'Slope (std)', ax[1])
my_regplot(gradient, np.nanmean(np.where(grad_resid.data_resid != 0, grad_resid.data_resid, np.nan), axis=1),
           'Gradient', 'RMSE (mean)', ax[2])
f.subplots_adjust(wspace=0.5)
f.savefig(os.path.join(environment.figdir, 'path_traversal_rlfp.png'),
          dpi=150, bbox_inches='tight', pad_inches=0.1)
plt.close()

# %% 3) ct
A.get_gradient_slopes(gradient, x_map=ct_mean.data_resid)
indices = np.where(~np.eye(n_parcels, dtype=bool) * ~np.isnan(A.grad_resid))

grad_slope = DataMatrix(data=A.grad_slope)
grad_slope.regress_nuisance(c=A.hops)
grad_resid = DataMatrix(data=A.grad_resid)
grad_resid.regress_nuisance(c=A.hops)

stats = sp.stats.pearsonr(A.grad_slope[indices], A.hops[indices])
print('Correlation between slope and hops: {:.2f}, {:.2f}'.format(stats[0], stats[1]))
stats = sp.stats.pearsonr(grad_slope.data_resid[indices], A.hops[indices])
print('\t after regressing out hops: {:.2f}, {:.2f}'.format(stats[0], stats[1]))

stats = sp.stats.pearsonr(A.grad_resid[indices], A.hops[indices])
print('Correlation between error and hops: {:.2f}, {:.2f}'.format(stats[0], stats[1]))
stats = sp.stats.pearsonr(grad_resid.data_resid[indices], A.hops[indices])
print('\t after regressing out hops: {:.2f}, {:.2f}'.format(stats[0], stats[1]))

f, ax = plt.subplots(3, 1, figsize=(5, 5*3))
my_regplot(gradient, np.nanmean(np.where(grad_slope.data_resid != 0, grad_slope.data_resid, np.nan), axis=1),
           'Gradient', 'Slope (mean)', ax[0])
my_regplot(gradient, np.nanstd(np.where(grad_slope.data_resid != 0, grad_slope.data_resid, np.nan), axis=1),
           'Gradient', 'Slope (std)', ax[1])
my_regplot(gradient, np.nanmean(np.where(grad_resid.data_resid != 0, grad_resid.data_resid, np.nan), axis=1),
           'Gradient', 'RMSE (mean)', ax[2])
f.subplots_adjust(wspace=0.5)
f.savefig(os.path.join(environment.figdir, 'path_traversal_ct.png'),
          dpi=150, bbox_inches='tight', pad_inches=0.1)
plt.close()

# %% 4) sst - pvalb
A.get_gradient_slopes(gradient, x_map=sst_pvalb_delta.data_resid)
indices = np.where(~np.eye(n_parcels, dtype=bool) * ~np.isnan(A.grad_resid))

grad_slope = DataMatrix(data=A.grad_slope)
grad_slope.regress_nuisance(c=A.hops)
grad_resid = DataMatrix(data=A.grad_resid)
grad_resid.regress_nuisance(c=A.hops)

stats = sp.stats.pearsonr(A.grad_slope[indices], A.hops[indices])
print('Correlation between slope and hops: {:.2f}, {:.2f}'.format(stats[0], stats[1]))
stats = sp.stats.pearsonr(grad_slope.data_resid[indices], A.hops[indices])
print('\t after regressing out hops: {:.2f}, {:.2f}'.format(stats[0], stats[1]))

stats = sp.stats.pearsonr(A.grad_resid[indices], A.hops[indices])
print('Correlation between error and hops: {:.2f}, {:.2f}'.format(stats[0], stats[1]))
stats = sp.stats.pearsonr(grad_resid.data_resid[indices], A.hops[indices])
print('\t after regressing out hops: {:.2f}, {:.2f}'.format(stats[0], stats[1]))

f, ax = plt.subplots(3, 1, figsize=(5, 5*3))
my_regplot(gradient, np.nanmean(np.where(grad_slope.data_resid != 0, grad_slope.data_resid, np.nan), axis=1),
           'Gradient', 'Slope (mean)', ax[0])
my_regplot(gradient, np.nanstd(np.where(grad_slope.data_resid != 0, grad_slope.data_resid, np.nan), axis=1),
           'Gradient', 'Slope (std)', ax[1])
my_regplot(gradient, np.nanmean(np.where(grad_resid.data_resid != 0, grad_resid.data_resid, np.nan), axis=1),
           'Gradient', 'RMSE (mean)', ax[2])
f.subplots_adjust(wspace=0.5)
f.savefig(os.path.join(environment.figdir, 'path_traversal_sst-pvalb.png'),
          dpi=150, bbox_inches='tight', pad_inches=0.1)
plt.close()
