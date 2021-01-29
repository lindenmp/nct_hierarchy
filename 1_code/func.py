# Linden Parkes, 2020
# lindenmp@seas.upenn.edu

# Essentials
import os, sys, glob
import pandas as pd
import numpy as np
import nibabel as nib

# Stats
import scipy as sp
from scipy import stats
import statsmodels.api as sm
import pingouin as pg

# Plotting
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'

# Extra
import math
from scipy.linalg import svd, schur


def set_proj_env():

    # Project root directory
    projdir = '/Users/lindenmp/Google-Drive-Penn/work/research_projects/pfactor_gradients'
    os.environ['PROJDIR'] = projdir

    # Data directory
    datadir = os.path.join(projdir, '0_data')
    os.environ['DATADIR'] = datadir

    # Imaging derivatives
    derivsdir = os.path.join('/Volumes/work_ssd/research_data/PNC/')
    os.environ['DERIVSDIR'] = derivsdir

    # Pipeline directory
    pipelinedir = os.path.join(projdir, '2_pipeline')
    os.environ['PIPELINEDIR'] = pipelinedir

    # Output directory
    outputdir = os.path.join(projdir, '3_output')
    os.environ['OUTPUTDIR'] = outputdir

    return projdir, datadir, derivsdir, pipelinedir, outputdir
    # return parcel_names, parcel_loc, drop_parcels, num_parcels, yeo_idx, yeo_labels


def my_get_cmap(which_type = 'qual1', num_classes = 8):
    # Returns a nice set of colors to make a nice colormap using the color schemes
    # from http://colorbrewer2.org/
    #
    # The online tool, colorbrewer2, is copyright Cynthia Brewer, Mark Harrower and
    # The Pennsylvania State University.

    if which_type == 'linden':
        cmap_base = np.array([[255,105,97],[97,168,255],[178,223,138],[117,112,179],[255,179,71]])
    elif which_type == 'pair':
        cmap_base = np.array([[124,230,199],[255,169,132]])
    elif which_type == 'qual1':
        cmap_base = np.array([[166,206,227],[31,120,180],[178,223,138],[51,160,44],[251,154,153],[227,26,28],
                            [253,191,111],[255,127,0],[202,178,214],[106,61,154],[255,255,153],[177,89,40]])
    elif which_type == 'qual2':
        cmap_base = np.array([[141,211,199],[255,255,179],[190,186,218],[251,128,114],[128,177,211],[253,180,98],
                            [179,222,105],[252,205,229],[217,217,217],[188,128,189],[204,235,197],[255,237,111]])
    elif which_type == 'seq_red':
        cmap_base = np.array([[255,245,240],[254,224,210],[252,187,161],[252,146,114],[251,106,74],
                            [239,59,44],[203,24,29],[165,15,21],[103,0,13]])
    elif which_type == 'seq_blu':
        cmap_base = np.array([[247,251,255],[222,235,247],[198,219,239],[158,202,225],[107,174,214],
                            [66,146,198],[33,113,181],[8,81,156],[8,48,107]])
    elif which_type == 'redblu_pair':
        cmap_base = np.array([[222,45,38],[49,130,189]])
    elif which_type == 'yeo17':
        cmap_base = np.array([[97,38,107], # VisCent
                            [194,33,39], # VisPeri
                            [79,130,165], # SomMotA
                            [44,181,140], # SomMotB
                            [75,148,72], # DorsAttnA
                            [23,116,62], # DorsAttnB
                            [149,77,158], # SalVentAttnA
                            [222,130,177], # SalVentAttnB
                            [75,87,61], # LimbicA
                            [149,166,110], # LimbicB
                            [210,135,47], # ContA
                            [132,48,73], # ContB
                            [92,107,131], # ContC
                            [218,221,50], # DefaultA
                            [175,49,69], # DefaultB
                            [41,38,99], # DefaultC
                            [53,75,158] # TempPar
                            ])
    elif which_type == 'yeo7':
        cmap_base = np.array([[162,81,172], # visual
                            [120,154,192], # somatomotor
                            [63,153,50], # dorsalAttention
                            [223,101,255], # salienceVentralAttention
                            [247,253,201], # limbic
                            [241,185,68], # frontoparietalControl
                            [217,112,123]]) # default

    if cmap_base.shape[0] > num_classes: cmap = cmap_base[0:num_classes]
    else: cmap = cmap_base

    cmap = cmap / 255

    return cmap


def roi_to_vtx(roi_data, parcel_names, parc_file):
    # roi_data      = (num_nodes,) array vector of node-level data to plot onto surface
    # 
    # 
    # parcel_names     = (num_nodes,) array vector of strings containg roi names
    #               corresponding to roi_data
    # 
    # parc_file    = full path and file name to surface file
    #               Note, I used fsaverage/fsaverage5 surfaces

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


def rank_to_normal(rank, c, n):
    # Standard quantile function
    x = (rank - c) / (n - 2*c + 1)
    return sp.stats.norm.ppf(x)


def rank_int(series, c=3.0/8):
    # Check input
    assert(isinstance(series, pd.Series))
    assert(isinstance(c, float))

    # Set seed
    np.random.seed(123)

    # Drop NaNs
    series = series.loc[~pd.isnull(series)]

    # Get rank, ties are averaged
    rank = sp.stats.rankdata(series, method="average")

    # Convert numpy array back to series
    rank = pd.Series(rank, index=series.index)

    # Convert rank to normal distribution
    transformed = rank.apply(rank_to_normal, c=c, n=len(rank))
    
    return transformed


def get_fdr_p(p_vals, alpha = 0.05):
    out = multitest.multipletests(p_vals, alpha = alpha, method = 'fdr_bh')
    p_fdr = out[1] 

    return p_fdr


def get_fdr_p_df(p_vals, alpha = 0.05, rows = False):
    
    if rows:
        p_fdr = pd.DataFrame(index = p_vals.index, columns = p_vals.columns)
        for row, data in p_vals.iterrows():
            p_fdr.loc[row,:] = get_fdr_p(data.values)
    else:
        p_fdr = pd.DataFrame(index = p_vals.index,
                            columns = p_vals.columns,
                            data = np.reshape(get_fdr_p(p_vals.values.flatten(), alpha = alpha), p_vals.shape))

    return p_fdr


def get_exact_p(x,y):
    pval = 2*np.min([np.mean(x-y>=0), np.mean(x-y<=0)])
    
    return pval


def node_strength(A):
    s = np.sum(A, axis = 0)

    return s


def minimum_energy(A, T, B, x0, xf):
    #  Computes minimum control energy for state transition.
    #  A: System adjacency matrix:         N x N
    #  B: Control input matrix:            N x k
    #  x0: Initial state:                  N x 1
    #  xf: Final state:                    N x 1
    #  T: Control horizon                  1 x 1
    #  
    #  Outputs
    #  x: State Trajectory
    #  u: Control Input

    # Author: Jennifer Stiso

    # System Size
    n = A.shape[0] # Number of nodes
    
    u, s, vt = svd(A) # singluar value decomposition
    A = A/(1 + s[0]) # Matrix normalization 

    # Compute Matrix Exponential
    AT = np.concatenate((np.concatenate((A, -.5*(B.dot(B.T))), axis=1), 
                         np.concatenate((np.zeros(np.shape(A)), -A.T), axis=1)), axis=0)

    E = sp.linalg.expm(AT*T)

    # Compute Costate Initial Condition
    E12 = E[0:n,n:]
    E11 = E[0:n,0:n]
    p0 = np.linalg.pinv(E12).dot(xf - E11.dot(x0))

    # Compute Costate Initial Condition Error Induced by Inverse
    n_err = np.linalg.norm(E12.dot(p0) - (xf - E11.dot(x0)))

    # Prepare Simulation
    nStep=1000
    t = np.linspace(0,T,nStep+1)

    v0 = np.concatenate((x0, p0), axis=0)   # Initial Condition
    v = np.zeros((2*n,len(t)))              # Trajectory
    Et = sp.linalg.expm(AT*T/(len(t)-1))
    v[:,0] = v0.T

    # Simulate State and Costate Trajectories
    for i in np.arange(1,len(t)):
        v[:,i] = Et.dot(v[:,i-1])

    x = v[0:n,:];
    u = -0.5*B.T.dot(v[np.arange(0,n)+n,:])

    # transpose to be similar to opt_eng_cont
    u = u.T
    x = x.T

    return x, u, n_err

