import os
import numpy as np
import nibabel as nib
import math

import scipy as sp
from scipy import stats

import seaborn as sns

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


def my_regplot(x, y, xlabel, ylabel, ax, c='gray'):
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

    color_blue = sns.color_palette("Set1")[1]
    try:
        sns.kdeplot(x=x, y=y, ax=ax, color='gray', thresh=0.05, alpha=0.25)
    except:
        pass
    sns.regplot(x=x, y=y, ax=ax, scatter=False, color=color_blue)
    ax.scatter(x=x, y=y, c=c, s=5, alpha=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel, labelpad=-1)
    ax.tick_params(pad=-2.5)
    ax.grid(False)
    sns.despine(right=True, top=True, ax=ax)
    pearson_stats = sp.stats.pearsonr(x, y)
    spearman_stats = sp.stats.spearmanr(x, y)
    textstr = 'r = {:.2f}, p = {:.2f} \nrho = {:.2f}, p = {:.2f}'.format(pearson_stats[0], pearson_stats[1],
                                                                         spearman_stats[0], spearman_stats[1])
    ax.text(0.05, 0.975, textstr, transform=ax.transAxes,
            verticalalignment='top')

def my_rnullplot(x, x_null, y, xlabel, ax):
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

def my_nullplot(observed, null, p_val, xlabel, ax):
    color_blue = sns.color_palette("Set1")[1]
    color_red = sns.color_palette("Set1")[0]
    sns.histplot(x=null, ax=ax, color='gray')
    ax.axvline(x=observed, ymax=1, clip_on=False, linewidth=1.5, color=color_blue)
    ax.grid(False)
    sns.despine(right=True, top=True, ax=ax)
    ax.tick_params(pad=-2.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('counts')

    textstr = 'observed = {:.2f}'.format(observed)
    ax.text(observed, ax.get_ylim()[1], textstr,
            horizontalalignment='left', verticalalignment='top', rotation=270, c=color_blue)

    textstr = 'p = {:.2f}'.format(p_val)
    ax.text(observed, ax.get_ylim()[1], textstr,
            horizontalalignment='right', verticalalignment='top', rotation=270, c=color_red)

