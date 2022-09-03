import sys, os, platform
from src.utils import get_p_val_string, get_exact_p

import numpy as np
import pandas as pd
import scipy as sp
import nibabel as nib
import math
from scipy import stats

import seaborn as sns
import pkg_resources
import matplotlib as mpl
import matplotlib.pyplot as plt


def set_plotting_params(format='png'):
    if platform.system() == 'Darwin':
        os.system('rm -rf ~/.cache/matplotlib')

    sns.set(style='whitegrid', context='paper', font_scale=1, font='Helvetica')

    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['savefig.format'] = format
    plt.rcParams['font.size'] = 8
    plt.rcParams['svg.fonttype'] = 'none'


def roi_to_vtx(roi_data, annot_file):
    labels, ctab, surf_names = nib.freesurfer.read_annot(annot_file)
    vtx_data = np.zeros(labels.shape)

    unique_labels = np.unique(labels)
    if unique_labels[0] == 0:
        unique_labels = unique_labels[1:]

    for i in unique_labels:
        vtx_data[labels == i] = roi_data[i - 1]

    # get min/max for plottin
    x = np.sort(np.unique(vtx_data))

    if x.shape[0] > 1:
        vtx_data_min = x[0]
        vtx_data_max = x[-1]
    else:
        vtx_data_min = 0
        vtx_data_max = 0

    return vtx_data, vtx_data_min, vtx_data_max


def my_reg_plot(x, y, xlabel, ylabel, ax, c='gray', annotate='pearson', bonferroni=False, regr_line=True, kde=True,
                fontsize=8):
    if len(x.shape) > 1 and len(y.shape) > 1:
        if x.shape[0] == x.shape[1] and y.shape[0] == y.shape[1]:
            mask_x = ~np.eye(x.shape[0], dtype=bool) * ~np.isnan(x)
            mask_y = ~np.eye(y.shape[0], dtype=bool) * ~np.isnan(y)
            mask = mask_x * mask_y
            indices = np.where(mask)
        else:
            mask_x = ~np.isnan(x)
            mask_y = ~np.isnan(y)
            mask = mask_x * mask_y
            indices = np.where(mask)
    elif len(x.shape) == 1 and len(y.shape) == 1:
        mask_x = ~np.isnan(x)
        mask_y = ~np.isnan(y)
        mask = mask_x * mask_y
        indices = np.where(mask)
    else:
        print('error: input array dimension mismatch.')

    try:
        x = x[indices]
        y = y[indices]
    except:
        pass

    try:
        c = c[indices]
    except:
        pass

    # kde plot
    if kde == True:
        try:
            sns.kdeplot(x=x, y=y, ax=ax, color='gray', thresh=0.05, alpha=0.25)
        except:
            pass

    # regression line
    if regr_line == True:
        color_blue = sns.color_palette("Set1")[1]
        sns.regplot(x=x, y=y, ax=ax, scatter=False, color=color_blue)

    # scatter plot
    if type(c) == str:
        ax.scatter(x=x, y=y, c=c, s=5, alpha=0.5)
    else:
        ax.scatter(x=x, y=y, c=c, cmap='viridis', s=5, alpha=0.5)

    # axis options
    ax.set_xlabel(xlabel, labelpad=-0.5)
    ax.set_ylabel(ylabel, labelpad=-0.5)
    ax.tick_params(pad=-2.5)
    ax.grid(False)
    sns.despine(right=True, top=True, ax=ax)

    # annotation
    r, r_p = sp.stats.pearsonr(x, y)
    rho, rho_p = sp.stats.spearmanr(x, y)
    if bonferroni != False:
        r_p = r_p / bonferroni
        rho_p = rho_p / bonferroni
    if annotate == 'pearson':
        textstr = '$\mathit{:}$ = {:.2f}, {:}'.format('{r}', r, get_p_val_string(r_p))
        ax.text(0.05, 0.975, textstr, transform=ax.transAxes, fontsize=fontsize,
                verticalalignment='top')
    elif annotate == 'spearman':
        textstr = '$\\rho$ = {:.2f}, {:}'.format(rho, get_p_val_string(rho_p))
        ax.text(0.05, 0.975, textstr, transform=ax.transAxes, fontsize=fontsize,
                verticalalignment='top')
    elif annotate == 'both':
        textstr = '$\mathit{:}$ = {:.2f}, {:}\n$\\rho$ = {:.2f}, {:}'.format('{r}', r, get_p_val_string(r_p),
                                                                             rho, get_p_val_string(rho_p))
        ax.text(0.05, 0.975, textstr, transform=ax.transAxes, fontsize=fontsize,
                verticalalignment='top')
    else:
        pass


def my_rnull_plot(x, x_null, y, xlabel, ax):
    if len(x.shape) > 1 and len(y.shape) > 1:
        if x.shape[0] == x.shape[1] and y.shape[0] == y.shape[1]:
            mask_x = ~np.eye(x.shape[0], dtype=bool) * ~np.isnan(x)
            mask_y = ~np.eye(y.shape[0], dtype=bool) * ~np.isnan(y)
            mask = mask_x * mask_y
            indices = np.where(mask)

            x = x[indices]
            y = y[indices]
            x_null = x_null[indices]

    num_surrogates = x_null.shape[1]

    y_null = np.zeros(num_surrogates)
    for i in np.arange(num_surrogates): y_null[i] = sp.stats.pearsonr(x_null[:, i], y)[0]

    sns.histplot(y_null, ax=ax)

    y_obs = sp.stats.pearsonr(x, y)[0]
    ax.axvline(x=y_obs, c='r')

    if y_obs < 0:
        p_val = np.sum(y_null <= y_obs) / num_surrogates
    else:
        p_val = np.sum(y_null >= y_obs) / num_surrogates
    # p_val = np.sum(np.abs(y_null) >= np.abs(y_obs))/num_surrogates
    # p_val = np.min([np.sum(y_null >= y_obs)/num_surrogates,
    #                 np.sum(y_null <= y_obs)/num_surrogates])
    textstr = 'p unc. = {:.2f}'.format(p_val)
    ax.text(0.01, 0.975, textstr, transform=ax.transAxes,
            verticalalignment='top', rotation='horizontal', c='r')

    ax.set_xlabel(xlabel)
    ax.set_ylabel('')
    ax.tick_params(pad=-2.5)


def my_null_plot(observed, null, p_val, xlabel, ax, fontsize=8):
    color_blue = sns.color_palette("Set1")[1]
    color_red = sns.color_palette("Set1")[0]
    sns.histplot(x=null, ax=ax, color='gray')
    ax.axvline(x=observed, ymax=1, clip_on=False, linewidth=1, color=color_blue)
    ax.grid(False)
    sns.despine(right=True, top=True, ax=ax)
    ax.tick_params(pad=-2.5)
    ax.set_xlabel(xlabel, labelpad=-0.5)
    ax.set_ylabel('counts', labelpad=-0.5)

    textstr = 'obs. = {:.2f}'.format(observed)
    ax.text(observed, ax.get_ylim()[1], textstr, fontsize=fontsize,
            horizontalalignment='left', verticalalignment='top', rotation=270, c=color_blue)

    textstr = '{:}'.format(get_p_val_string(p_val))
    ax.text(observed - (np.abs(observed)*0.0025), ax.get_ylim()[1], textstr, fontsize=fontsize,
            horizontalalignment='right', verticalalignment='top', rotation=270, c=color_red)


def my_distpair_plot(df, ylabel, ax, test_stat='ttest_1samp', fontsize=8):
    sns.violinplot(data=df, ax=ax, inner="box", palette="pastel", cut=2, linewidth=1.5)

    sns.despine(left=True, bottom=True)
    ax.set_ylabel(ylabel, labelpad=-0.5)
    ax.tick_params(pad=-2.5)

    if test_stat == 'exact':
        p_val = get_exact_p(df.iloc[:, 0], df.iloc[:, 1])
        textstr = get_p_val_string(p_val)
        ax.text(0.5, ax.get_ylim()[1], textstr, fontsize=fontsize,
                horizontalalignment='center', verticalalignment='bottom')
        ax.axhline(y=ax.get_ylim()[1], xmin=0.25, xmax=0.75, color='k', linewidth=1)
    elif test_stat == 'ttest':
        t, p_val = sp.stats.ttest_rel(a=df.iloc[:, 0], b=df.iloc[:, 1])
        textstr = '$\mathit{:}$ = {:.2f}, {:}'.format('{t}', t, get_p_val_string(p_val))
        ax.text(0.5, ax.get_ylim()[1], textstr, fontsize=fontsize,
                horizontalalignment='center', verticalalignment='bottom')
        ax.axhline(y=ax.get_ylim()[1], xmin=0.25, xmax=0.75, color='k', linewidth=1)
    elif test_stat == 'ttest_1samp':
        t, p_val = sp.stats.ttest_1samp(df.iloc[:, 0] - df.iloc[:, 1], popmean=0)
        textstr = '$\mathit{:}$ = {:.2f}, {:}'.format('{t}', t, get_p_val_string(p_val))
        ax.text(0.5, ax.get_ylim()[1], textstr, fontsize=fontsize,
                horizontalalignment='center', verticalalignment='bottom')
        ax.axhline(y=ax.get_ylim()[1], xmin=0.25, xmax=0.75, color='k', linewidth=1)
    elif test_stat == 'mean_diff':
        stat = np.mean(df.iloc[:, 0] - df.iloc[:, 1])
        textstr = 'mean diff = {:.4f}'.format(stat)
        ax.text(0.5, ax.get_ylim()[1], textstr, fontsize=fontsize,
                horizontalalignment='center', verticalalignment='bottom')
        ax.axhline(y=ax.get_ylim()[1], xmin=0.25, xmax=0.75, color='k', linewidth=1)
    elif test_stat is None:
        pass


def my_bsci_plot(dist, observed, xlabel, ax, fontsize=8):
    color_blue = sns.color_palette("Set1")[1]
    color_red = sns.color_palette("Set1")[0]
    conf_interval = np.percentile(dist, [2.5, 97.5])
    # observed = np.mean(dist)

    sns.histplot(x=dist, ax=ax, color='gray')
    ax.axvline(conf_interval[0], ymax=1, clip_on=False, linewidth=1, color=color_red)
    ax.axvline(observed, ymax=1, clip_on=False, linewidth=1, color=color_blue)
    ax.axvline(conf_interval[1], ymax=1, clip_on=False, linewidth=1, color=color_red)
    ax.grid(False)
    sns.despine(right=True, top=True, ax=ax)
    ax.tick_params(pad=-2.5)
    ax.set_xlabel(xlabel, labelpad=-0.5)
    ax.set_ylabel('counts', labelpad=-0.5)

    textstr = 'lower = {:.4f}'.format(conf_interval[0])
    ax.text(conf_interval[0], ax.get_ylim()[1], textstr, fontsize=fontsize,
            horizontalalignment='left', verticalalignment='top', rotation=270, c=color_red)

    textstr = 'obs. = {:.4f}'.format(observed)
    ax.text(observed, ax.get_ylim()[1], textstr, fontsize=fontsize,
            horizontalalignment='left', verticalalignment='top', rotation=270, c=color_blue)

    textstr = 'upper = {:.4f}'.format(conf_interval[1])
    ax.text(conf_interval[1], ax.get_ylim()[1], textstr, fontsize=fontsize,
            horizontalalignment='left', verticalalignment='top', rotation=270, c=color_red)


def my_bsci_pair_plot(dist1, observed1, dist2, observed2, xlabel, ax, fontsize=8):
    color_blue = sns.color_palette("Set1")[1]
    color_red = sns.color_palette("Set1")[0]

    # dist 1
    sns.histplot(x=dist1, ax=ax, color='black', kde=False, alpha=0.2)
    conf_interval = np.percentile(dist1, [2.5, 97.5])
    ax.axvline(conf_interval[0], ymax=1, clip_on=False, linewidth=1, color=color_red)
    ax.axvline(observed1, ymax=1, clip_on=False, linewidth=1, color=color_blue)
    ax.axvline(conf_interval[1], ymax=1, clip_on=False, linewidth=1, color=color_red)

    textstr = 'lower = {:.4f}'.format(conf_interval[0])
    ax.text(conf_interval[0], ax.get_ylim()[1], textstr, fontsize=fontsize,
            horizontalalignment='left', verticalalignment='top', rotation=270, c=color_red)

    textstr = 'obs. = {:.4f}'.format(observed1)
    ax.text(observed1, ax.get_ylim()[1], textstr, fontsize=fontsize,
            horizontalalignment='left', verticalalignment='top', rotation=270, c=color_blue)

    textstr = 'upper = {:.4f}'.format(conf_interval[1])
    ax.text(conf_interval[1], ax.get_ylim()[1], textstr, fontsize=fontsize,
            horizontalalignment='left', verticalalignment='top', rotation=270, c=color_red)

    # dist 2
    sns.histplot(x=dist2, ax=ax, color='gray', kde=False)
    conf_interval = np.percentile(dist2, [2.5, 97.5])
    ax.axvline(conf_interval[0], ymax=1, clip_on=False, linewidth=1, color=color_red)
    ax.axvline(observed2, ymax=1, clip_on=False, linewidth=1, color=color_blue)
    ax.axvline(conf_interval[1], ymax=1, clip_on=False, linewidth=1, color=color_red)

    textstr = 'lower = {:.4f}'.format(conf_interval[0])
    ax.text(conf_interval[0], ax.get_ylim()[1], textstr, fontsize=fontsize,
            horizontalalignment='left', verticalalignment='top', rotation=270, c=color_red)

    textstr = 'obs. = {:.4f}'.format(observed2)
    ax.text(observed2, ax.get_ylim()[1], textstr, fontsize=fontsize,
            horizontalalignment='left', verticalalignment='top', rotation=270, c=color_blue)

    textstr = 'upper = {:.4f}'.format(conf_interval[1])
    ax.text(conf_interval[1], ax.get_ylim()[1], textstr, fontsize=fontsize,
            horizontalalignment='left', verticalalignment='top', rotation=270, c=color_red)

    ax.grid(False)
    sns.despine(right=True, top=True, ax=ax)
    ax.tick_params(pad=-2.5)
    ax.set_xlabel(xlabel, labelpad=-0.5)
    ax.set_ylabel('counts', labelpad=-0.5)
