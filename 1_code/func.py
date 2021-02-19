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
from statsmodels.stats import multitest
from scipy.spatial.distance import pdist, squareform
from bct.utils import weight_conversion
from bct.algorithms.distance import distance_wei, distance_wei_floyd, retrieve_shortest_path

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


def minimum_energy(A, T, B, x0, xf, c = 1):
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
    A = A/(c + s[0]) - np.eye(A.shape[0]) # Matrix normalization 

    if type(x0[0]) == np.bool_:
        x0 = x0.astype(float)
    if type(xf[0]) == np.bool_:
        xf = xf.astype(float)

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


def minimum_energy_nonh(A, T, B, x0, xf, c = 1):
    u, s, vt = svd(A) # singluar value decomposition
    A = A/(c + s[0]) - np.eye(A.shape[0]) # Matrix normalization 

    Phi = np.dot(sp.linalg.expm(A*T), x0) - xf

    Wc = np.zeros(A.shape)
    nN = 1000
    for i in np.arange(0,nN):
        FM = np.matmul(sp.linalg.expm(A*T*(i/nN)),B)
        Wc += np.matmul(FM,FM.T)*(T/nN)

    E = np.dot(Phi.T,sp.linalg.solve(Wc,Phi))
    
    return E


def get_time_vec(T, num_taylor):
    return np.power(T, np.arange(0,num_taylor))


def minimum_energy_taylor(A, T, B, x0, xf, c = 1, num_taylor = 10, drop_taylor = 0):
    num_parcels = A.shape[0] # Number of nodes

    u, s, vt = svd(A) # singluar value decomposition
    A = A/(c + s[0]) - np.eye(A.shape[0]) # Matrix normalization 
    
    # Define Taylor series coefficients
    tc = np.zeros(num_taylor)
    for i in np.arange(0,num_taylor):
        tc[i] = 1/math.factorial(i)
        
    if drop_taylor > 0:
        tc[drop_taylor] = 0

    # Define matrices
    AM = np.zeros((num_parcels,num_parcels,num_taylor)) # Matrix powers along 3rd dimension
    AM[:,:,0] = np.eye(num_parcels)
    for i in np.arange(1,num_taylor):
        AM[:,:,i] = np.matmul(AM[:,:,i-1], A)

    # Combine matrices and coefficients
    AM = np.multiply(AM,tc);

    # Define state transition to achieve
    t_vec = get_time_vec(T, num_taylor)
    Phi = np.dot(np.sum(np.multiply(AM, t_vec),2), x0) - xf

    # Gramian
    Wc = np.zeros(A.shape) # Initialize Gramian
    nN = 1000 # Number of integration steps
    for i in np.arange(0,nN):
        t_vec = get_time_vec(T * (i/nN), num_taylor)
        t_vec = t_vec.reshape(num_taylor,1)
        FM = np.matmul(np.squeeze(np.dot(AM, t_vec)), B)
        Wc += np.matmul(FM,FM.T)*(T/nN)

    E = np.dot(Phi.T,sp.linalg.solve(Wc,Phi))
    
    return E


def optimal_energy(A, T, B, x0, xf, rho, S, c = 1):
    # This is a python adaptation of matlab code originally written by Tomaso Menara and Jason Kim
    #% compute optimal inputs/trajectories
    #% Fabio, Tommy September 2017
    #%
    #% -------------- Change Log -------------
    #% JStiso April 2018
    #%   Changed S to be an input, rather than something defined internally
    #%
    #% Jason Kim January 2021
    #%   Changed the forward propagation of states to matrix exponential to
    #%   avoid reliance on MATLAB toolboxes. Also changed definition of expanded
    #%   input U to save time by avoiding having to resize the matrix.
    # %   Also changed the initialization of U_opt for the same reason.
    #
    # JStiso 2021
    #     Translated to Python

    # % Inputs:
    # % A     (NxN numpy array) Structural connectivity matrix
    # % B     (NxN numpy array) Input matrix: selects which nodes to put input into. Define
    # %       so there is a 1 on the diagonal of elements you want to add input to, 
    # %       and 0 otherwise 
    # % S     (NxN numpy array) Selects nodes whose distance you want to constrain, Define so
    # %       that there is a 1 on the diagonal of elements you want to
    # %       constrain, and a zero otherwise
    # % T     (float) Time horizon: how long you want to control for. Too large will give
    # %       large error, too short will not give enough time for control
    # % rho   (float) weights energy and distance constraints. Small rho leads to larger
    # %       energy
    #
    # Outputs:
    # X_opt    (TxN numpy array) The optimal trajectory through state space
    # U_opt    (TxN numpy array) The optimal energy
    # n_err    (float) the error associated with this calculation. Errors will be larger when B is not identity, 
    #          and when A is large. Large T and rho will also tend to increase the error

    n = A.shape[0] # Number of nodes

    u, s, vt = svd(A) # singluar value decomposition
    A = A/(c + s[0]) - np.eye(A.shape[0]) # Matrix normalization 

    if type(x0[0]) == np.bool_:
        x0 = x0.astype(float)
    if x0.ndim == 1:
        x0 = x0.reshape(-1,1)

    if type(xf[0]) == np.bool_:
        xf = xf.astype(float)
    if xf.ndim == 1:
        xf = xf.reshape(-1,1)

    Sbar = np.eye(n) - S
    np.shape(np.dot(-B,B.T)/(2*rho))

    Atilde = np.concatenate((np.concatenate((A, np.dot(-B,B.T)/(2*rho)), axis=1), 
                            np.concatenate((-2*S, -A.T), axis=1)), axis=0)

    M = sp.linalg.expm(Atilde*T)
    M11 = M[0:n,0:n]
    M12 = M[0:n,n:]
    M21 = M[n:,0:n]
    M22 = M[n:,n:]

    N = np.linalg.solve(Atilde,(M-np.eye(np.shape(Atilde)[0])))
    c = np.dot(np.dot(N,np.concatenate((np.zeros((n,n)),S),axis = 0)),2*xf)
    c1 = c[0:n]
    c2 = c[n:]

    p0 = np.dot(np.linalg.pinv(np.concatenate((np.dot(S,M12),np.dot(Sbar,M22)), axis = 0)),
                        (-np.dot(np.concatenate((np.dot(S,M11),np.dot(Sbar,M21)),axis=0),x0) - 
                         np.concatenate((np.dot(S,c1),np.dot(Sbar,c2)), axis=0) + 
                         np.concatenate((np.dot(S,xf),np.zeros((n,1))), axis=0)))
    
    n_err = np.linalg.norm(np.dot(np.concatenate((np.dot(S,M12),np.dot(Sbar,M22)), axis = 0),p0) - 
                           (-np.dot(np.concatenate((np.dot(S,M11),np.dot(Sbar,M21)),axis=0),x0) - 
                            np.concatenate((np.dot(S,c1),np.dot(Sbar,c2)), axis=0) + 
                            np.concatenate((np.dot(S,xf),np.zeros((n,1))), axis=0))) # norm(error)

    STEP = 0.001
    t = np.arange(0,(T+STEP),STEP)

    U = np.dot(np.ones((np.size(t),1)),2*xf.T)

    # Discretize continuous-time input for convolution
    Atilde_d = sp.linalg.expm(Atilde*STEP)
    Btilde_d = np.linalg.solve(Atilde,
                               np.dot((Atilde_d-np.eye(2*n)),np.concatenate((np.zeros((n,n)),S), axis=0)))

    # Propagate forward discretized model
    xp = np.zeros((2*n,np.size(t)))
    xp[:,0:1] = np.concatenate((x0,p0), axis=0)
    for i in np.arange(1,np.size(t)):
        xp[:,i] = np.dot(Atilde_d,xp[:,i-1]) + np.dot(Btilde_d,U[i-1,:].T)

    xp = xp.T

    U_opt = np.zeros((np.size(t),np.shape(B)[1]))
    for i in range(np.size(t)):
        U_opt[i,:] = -(1/(2*rho))*np.dot(B.T,xp[i,n:].T)

    X_opt = xp[:,0:n]
    
    return X_opt, U_opt, n_err


def sample_gradient(gradient_2, gradient_1, seed = 'vis', num_step = 5, num_n = 10):
    
    num_parcels = len(gradient_2)
    
    # transmodal reference region
    ref_region = [gradient_2[gradient_1.argmax()], gradient_1.max()]
    
    if seed == 'vis':
        seed_region = [gradient_2.min(), gradient_1[gradient_2.argmin()]]
    elif seed == 'sm':
        seed_region = [gradient_2.max(), gradient_1[gradient_2.argmax()]]
        
    # create sampling space
    x = np.linspace(seed_region[0], ref_region[0], num_step)
    y = np.linspace(seed_region[1], ref_region[1], num_step)
    
    grad_coords = np.vstack([gradient_2.copy(), gradient_1.copy()]).transpose()
    gradient_masks = np.zeros((num_parcels,num_step)).astype(bool)
    
    for i in np.arange(0,num_step):
        xy = [x[i], y[i]]

        dist = (grad_coords - xy)**2
        dist = np.sum(dist, axis=1)
        dist = np.sqrt(dist)

        nn = np.argsort(dist)[:num_n]
        gradient_masks[nn,i] = True
        
        grad_coords[nn,:] = np.nan
        
    grad_coords = np.vstack([gradient_2.copy(), gradient_1.copy()]).transpose()
    dist_h = get_pdist(grad_coords,gradient_masks)
    
    return gradient_masks, dist_h


def get_pdist(coords, labels, method = 'mean'):
    unique, counts = np.unique(labels, return_counts = True)
    n_clusters = len(unique)
    
    dist = np.zeros((n_clusters,n_clusters))
    
    for i in np.arange(n_clusters):
        for j in np.arange(n_clusters):
            x0 = labels == i
            xf = labels == j
        
            x0_coords = coords[x0,:]
            xf_coords = coords[xf,:]
            
            tmp = []
            for r1 in np.arange(x0_coords.shape[0]):
                for r2 in np.arange(xf_coords.shape[0]):
                    d = (x0_coords[r1,:] - xf_coords[r2,:])**2
                    d = np.sum(d)
                    d = np.sqrt(d)
                    tmp.append(d)
            
            if method == 'mean':
                dist[i,j] = np.mean(tmp)
            elif method == 'min':
                dist[i,j] = np.min(tmp)
            elif method == 'median':
                dist[i,j] = np.median(tmp)
    
    return dist


def get_equidistant_state(x0, xf, dist_mni, centroids, num_n = 10):
    
    dist = squareform(pdist(centroids.values, 'euclidean')) # getting pairwise distance between regions MNI coords
    tolerance = np.std(dist)/2 # calculate tolerance for sampling (1/2 std of distances)
    # get lower and upper bounds for distance-based search
    lower_bound = dist_mni - tolerance
    upper_bound = dist_mni + tolerance
    
    dist = dist[xf.astype(bool),:] # retain only distance to n regions in the target target
    dist = np.nanmean(dist, axis = 0) # distance over n target state regions. dist is now vector
    dist[x0.astype(bool)] = np.nan # set distance for target state regions to nan. this prevents them from being sampled
    dist[xf.astype(bool)] = np.nan # set distance for initial state regions to nan. this prevents them from being sampled
    
    mask = (dist > lower_bound) & (dist < upper_bound) # create mask to sample from
    idx = np.random.choice(np.arange(0,np.sum(mask)), size = num_n, replace = False) # sample n regions from mask

    return np.array(centroids.iloc[mask,:].iloc[idx].index)-1 # return the indices of those regions


def get_B_matrix(x0, xf, control = 'wb'):
    num_parcels = x0.shape[0]
    
    if control == 'wb':
        B = np.eye(num_parcels)
    elif control == 'x0xf':
        B = np.zeros((num_parcels,num_parcels))
        B[x0,x0] = 1
        B[xf,xf] = 1
    elif control == 'x0':
        B = np.zeros((num_parcels,num_parcels))
        B[x0,x0] = 1
    elif control == 'xf':
        B = np.zeros((num_parcels,num_parcels))
        B[xf,xf] = 1
    elif control == 'x0xfwb':
        B = np.zeros((num_parcels,num_parcels))
        B[np.eye(num_parcels) == 1] = 5*10e-5
        B[x0,x0] = 1
        B[xf,xf] = 1
    elif control == 'x0wb':
        B = np.zeros((num_parcels,num_parcels))
        B[np.eye(num_parcels) == 1] = 5*10e-5
        B[x0,x0] = 1
    elif control == 'xfwb':
        B = np.zeros((num_parcels,num_parcels))
        B[np.eye(num_parcels) == 1] = 5*10e-5
        B[xf,xf] = 1

    return B


def subsample_state(x,subsample_size):
    x_tmp = np.zeros(x.size).astype(bool)
    
    sample = np.random.choice(np.where(x == True)[0], size = subsample_size, replace = False)
    
    x_tmp[sample] = True

    return x_tmp


def matrix_to_states(x, cluster_labels):
    
    unique, counts = np.unique(cluster_labels, return_counts = True)
    n_clusters = len(unique)
    
    x_out = np.zeros((n_clusters,n_clusters))
    for i in np.arange(n_clusters):
        for j in np.arange(n_clusters):
            x_out[i,j] = x[cluster_labels == i,:].mean(axis = 0)[cluster_labels == j].mean()
            
    return x_out


def get_gradient_variance(shortest_path, gradients, return_abs = False):

    # calculate the differences between coordinates in adjacent nodes along shortest path
    gradient_diff = np.diff(gradients[shortest_path,:], axis = 0)

    if return_abs == True:
         gradient_diff = np.abs(gradient_diff)

    mean_diff = np.mean(gradient_diff, axis = 0) # mean the differences along the shortest path
    var_diff = np.var(gradient_diff, axis = 0) # get the variance of the differences along the shortest path

    euclidean_var = np.var(np.sqrt(np.sum(np.square(gradient_diff), axis = 1))) # get the variance of the euclidean distance

    return mean_diff, var_diff, euclidean_var


def get_gradient_num_flips(shortest_path, gradients):

    # calculate the differences between coordinates in adjacent nodes along shortest path
    gradient_diff = np.diff(gradients[shortest_path,:], axis = 0)

    # get number of times there is a traversal direction reversal
    zero_crossings_g1 = np.where(np.diff(np.sign(gradient_diff[:,0])))[0]
    zero_crossings_g2 = np.where(np.diff(np.sign(gradient_diff[:,1])))[0]
    num_flips = np.array([len(zero_crossings_g1),len(zero_crossings_g2)])

    return num_flips


def get_adj_stats(A, gradients, cluster_labels, return_abs = False):

    num_parcels = A.shape[0]
    
    # convert to distance matrix
    D, hops, Pmat = distance_wei_floyd(A, transform = 'inv')
    
    # downsample distance matrix to cluster-based states
    D_mean = matrix_to_states(D, cluster_labels)
    hops_mean = matrix_to_states(hops, cluster_labels)
    
    # get transmodal and sensorimotor-visual traversal variance
    tm_tmp = np.zeros((num_parcels,num_parcels))
    tm_var_tmp = np.zeros((num_parcels,num_parcels))
    smv_tmp = np.zeros((num_parcels,num_parcels))
    smv_var_tmp = np.zeros((num_parcels,num_parcels))
    joint_var_tmp = np.zeros((num_parcels,num_parcels))
    num_tm_flips_tmp = np.zeros((num_parcels,num_parcels))
    num_smv_flips_tmp = np.zeros((num_parcels,num_parcels))

    for i in np.arange(num_parcels):
        for j in np.arange(num_parcels):
            shortest_path = retrieve_shortest_path(i,j,hops,Pmat)
            if len(shortest_path) != 0:
                shortest_path = shortest_path.flatten()
                mean_diff, var_diff, euclidean_var = get_gradient_variance(shortest_path, gradients, return_abs = return_abs)

                tm_tmp[i,j] = mean_diff[0]
                tm_var_tmp[i,j] = var_diff[0]

                smv_tmp[i,j] = mean_diff[1]
                smv_var_tmp[i,j] = var_diff[1]

                joint_var_tmp[i,j] = euclidean_var
                
                num_flips = get_gradient_num_flips(shortest_path, gradients)
                
                num_tm_flips_tmp[i,j] = num_flips[0]
                num_smv_flips_tmp[i,j] = num_flips[1]

    # downsample transmodal convergence to cluster-based states
    tm_con = matrix_to_states(tm_tmp, cluster_labels)
    tm_var = matrix_to_states(tm_var_tmp, cluster_labels)

    smv_con = matrix_to_states(smv_tmp, cluster_labels)
    smv_var = matrix_to_states(smv_var_tmp, cluster_labels)

    joint_var = matrix_to_states(joint_var_tmp, cluster_labels)
    
    num_tm_flips = matrix_to_states(num_tm_flips_tmp, cluster_labels)
    num_smv_flips = matrix_to_states(num_smv_flips_tmp, cluster_labels)
            
    return D_mean, hops_mean, tm_con, tm_var, smv_con, smv_var, joint_var, num_tm_flips, num_smv_flips

