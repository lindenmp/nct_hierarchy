import sys, os, platform
from pfactor_gradients.utils import get_p_val_string, get_exact_p, mean_confidence_interval

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
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['savefig.format'] = format
    plt.rcParams['font.size'] = 8

    # path = pkg_resources.resource_stream('pfactor_gradients', 'PublicSans-Thin.ttf')
    # prop = mpl.font_manager.FontProperties(fname=path.name)
    # plt.rcParams['font.sans-serif'] = prop.get_name()
    # plt.rcParams['font.serif'] = prop.get_name()
    # plt.rcParams['font.family'] = prop.get_family()
    # plt.rcParams['mathtext.fontset'] = 'custom'
    # plt.rcParams['mathtext.it'] = 'Public Sans:italic'
    # plt.rcParams['mathtext.bf'] = 'Public Sans:bold'
    # plt.rcParams['mathtext.cal'] = 'Public Sans'

    plt.rcParams['svg.fonttype'] = 'none'
    # sns.set(style='whitegrid', context='paper', font_scale=1, font='Public Sans')
    sns.set(style='whitegrid', context='paper', font_scale=1, font='Helvetica')


def roi_to_vtx(roi_data, parcel_names, parc_file):
    """
    Parameters
    ----------
    roi_data : np.array (n_parcels,)
        node-level data to plot onto surface
    parcel_names : np.array (n_parcels,)
        contains strings containing roi names corresponding to roi_data
    parc_file : str
        full path and file name to surface file.
        Note, I used fsaverage/fsaverage5 surfaces
    Returns
    -------
    vtx_data : np.array (n_vertices,)
        roi_data project onto sruface
    """

    # Load freesurfer file
    labels, ctab, surf_names = nib.freesurfer.read_annot(parc_file)

    # convert FS surf_names to array of strings
    if type(surf_names[0]) != str:
        for i in np.arange(0,len(surf_names)):
            surf_names[i] = surf_names[i].decode("utf-8")

    if 'myaparc' in parc_file:
        hemi = os.path.basename(parc_file)[0:2]

        # add hemisphere to surface surf_names
        for i in np.arange(0,len(surf_names)):
            surf_names[i] = hemi + "_" + surf_names[i]

    # Find intersection between parcel_names and surf_names
    overlap = np.intersect1d(parcel_names, surf_names, return_indices = True)
    overlap_names = overlap[0]
    idx_in = overlap[1] # location of surf_names in parcel_names
    idx_out = overlap[2] # location of parcel_names in surf_names

    # check for weird floating nans in roi_data
    fckn_nans = np.zeros((roi_data.shape)).astype(bool)
    for i in range(0,fckn_nans.shape[0]): fckn_nans[i] = math.isnan(roi_data[i])
    if any(fckn_nans): roi_data[fckn_nans] = 0

    # broadcast roi data to FS space
    # initialise idx vector with the dimensions of the FS labels, but data type corresponding to the roi data
    vtx_data = np.zeros(labels.shape, type(roi_data))
    # vtx_data = vtx_data - 1000

    # for each entry in fs names
    for i in range(0, overlap_names.shape[0]):
        vtx_data[labels == idx_out[i]] = roi_data[idx_in[i]]

    # get min/max for plottin
    x = np.sort(np.unique(vtx_data))

    if x.shape[0] > 1:
        vtx_data_min = x[0]
        vtx_data_max = x[-1]
    else:
        vtx_data_min = 0
        vtx_data_max = 0

    # i = 0
    # while vtx_data_min == -1000: vtx_data_min = x[i]; i += 1

    return vtx_data, vtx_data_min, vtx_data_max


def my_reg_plot(x, y, xlabel, ylabel, ax, c='gray', annotate='pearson', regr_line=True, kde=True):
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
    if annotate == 'pearson':
        textstr = '$\mathit{:}$ = {:.2f}, {:}'.format('{r}', r, get_p_val_string(r_p))
        ax.text(0.05, 0.975, textstr, transform=ax.transAxes, fontsize=8,
                verticalalignment='top')
    elif annotate == 'spearman':
        textstr = '$\\rho$ = {:.2f}, {:}'.format(rho, get_p_val_string(rho_p))
        ax.text(0.05, 0.975, textstr, transform=ax.transAxes, fontsize=8,
                verticalalignment='top')
    elif annotate == 'both':
        textstr = '$\mathit{:}$ = {:.2f}, {:}\n$\\rho$ = {:.2f}, {:}'.format('{r}', r, get_p_val_string(r_p),
                                                                             rho, get_p_val_string(rho_p))
        ax.text(0.05, 0.975, textstr, transform=ax.transAxes, fontsize=8,
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


def my_null_plot(observed, null, p_val, xlabel, ax):
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
    ax.text(observed, ax.get_ylim()[1], textstr, fontsize=8,
            horizontalalignment='left', verticalalignment='top', rotation=270, c=color_blue)

    textstr = '{:}'.format(get_p_val_string(p_val))
    ax.text(observed - (np.abs(observed)*0.0025), ax.get_ylim()[1], textstr, fontsize=8,
            horizontalalignment='right', verticalalignment='top', rotation=270, c=color_red)


def my_distpair_plot(df, ylabel, ax, test_stat='ttest', split=False):
    if split == True:
        mean_x1, lower_x1, upper_x1 = mean_confidence_interval(data=np.abs(df.iloc[:, 0]))
        mean_x2, lower_x2, upper_x2 = mean_confidence_interval(data=np.abs(df.iloc[:, 1]))

        df_tmp = pd.melt(df)
        df_tmp["x"] = ""
        sns.violinplot(data=df_tmp, ax=ax, x="x", y="value", hue="variable", inner=None, palette="pastel",
                       cut=2, linewidth=1.5, linecolor="k", split=split)
        ax.set_xlabel('')
        # ax.legend(title='', bbox_to_anchor=(1, 1.15), loc='upper right')
        ax.legend_.remove()

        # add point estimate with 95% CI
        ax.axhline(y=mean_x1, xmin=0.49, xmax=0.25, color="white", linewidth=1)
        ax.axhline(y=upper_x1, xmin=0.49, xmax=0.25, color="white", linewidth=0.5, linestyle=':')
        ax.axhline(y=lower_x1, xmin=0.49, xmax=0.25, color="white", linewidth=0.5, linestyle=':')
        ax.text(-0.15, upper_x1, '95% CI', fontsize=4,
                horizontalalignment='center', verticalalignment='bottom', rotation=0, c="white")

        ax.axhline(y=mean_x2, xmin=0.51, xmax=0.75, color="white", linewidth=1)
        ax.axhline(y=upper_x2, xmin=0.51, xmax=0.75, color="white", linewidth=0.5, linestyle=':')
        ax.axhline(y=lower_x2, xmin=0.51, xmax=0.75, color="white", linewidth=0.5, linestyle=':')
        ax.text(0.15, upper_x2, '95% CI', fontsize=4,
                horizontalalignment='center', verticalalignment='bottom', rotation=0, c="white")
    else:
        sns.violinplot(data=df, ax=ax, inner="box", palette="pastel", cut=2, linewidth=1.5)

    sns.despine(left=True, bottom=True)
    ax.set_ylabel(ylabel, labelpad=-0.5)
    ax.tick_params(pad=-2.5)

    if test_stat == 'exact':
        p_val = get_exact_p(df.iloc[:, 0], df.iloc[:, 1])
        textstr = get_p_val_string(p_val)
        ax.text(0.5, ax.get_ylim()[1], textstr, fontsize=8,
                horizontalalignment='center', verticalalignment='bottom')
        ax.axhline(y=ax.get_ylim()[1], xmin=0.25, xmax=0.75, color='k', linewidth=1)
    elif test_stat == 'ttest':
        t, p_val = sp.stats.ttest_rel(a=df.iloc[:, 0], b=df.iloc[:, 1])
        textstr = '$\mathit{:}$ = {:.2f}, {:}'.format('{t}', t, get_p_val_string(p_val))
        ax.text(0.5, ax.get_ylim()[1], textstr, fontsize=8,
                horizontalalignment='center', verticalalignment='bottom')
        ax.axhline(y=ax.get_ylim()[1], xmin=0.25, xmax=0.75, color='k', linewidth=1)
    elif test_stat is None:
        pass
