import argparse

import os, sys
import numpy as np
import scipy as sp

import numpy.matlib
import math
from scipy.linalg import svd
from sklearn.cluster import KMeans
from bct.algorithms.distance import retrieve_shortest_path, distance_wei_floyd

# --------------------------------------------------------------------------------------------------------------------
# parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument("-subjid", help="label for participant", dest="subjid", default=None, type=str)
parser.add_argument("-A_file", help="path and file to adjacency matrix", dest="A_file", default=None, type=str)
parser.add_argument("-outputdir", help="output directory", dest="outputdir", default=None, type=str)

parser.add_argument("-T", help="", dest="T", default=1, type=int)
parser.add_argument("-B_ver", help="", dest="B_ver", default='x0xfwb', type=str)
parser.add_argument("-n_subsamples", help="number of times a state is subsampled", dest="n_subsamples", default=50, type=int)
parser.add_argument("-n_taylor", help="number of taylor series to estimate", dest="n_taylor", default=7, type=int)
parser.add_argument("-drop_taylor_file", help="matrix for taylor polynomials to drop for each state transition", dest="drop_taylor_file", default=None, type=str)

parser.add_argument("-gradients_file", help="", dest="gradients_file", default=None, type=str)

args = parser.parse_args()

subjid = args.subjid
A_file = args.A_file
outputdir = args.outputdir

T = args.T
B_ver = args.B_ver
n_subsamples = args.n_subsamples
n_taylor = args.n_taylor
drop_taylor_file = args.drop_taylor_file

gradients_file = args.gradients_file

# --------------------------------------------------------------------------------------------------------------------
# functions
def get_B_matrix(x0, xf, version = 'wb'):
    num_parcels = x0.shape[0]
    
    if version == 'wb':
        B = np.eye(num_parcels)
    elif version == 'x0xf':
        B = np.zeros((num_parcels,num_parcels))
        B[x0,x0] = 1
        B[xf,xf] = 1
    elif version == 'x0':
        B = np.zeros((num_parcels,num_parcels))
        B[x0,x0] = 1
    elif version == 'xf':
        B = np.zeros((num_parcels,num_parcels))
        B[xf,xf] = 1
    elif version == 'x0xfwb':
        B = np.zeros((num_parcels,num_parcels))
        B[np.eye(num_parcels) == 1] = 5*10e-5
        B[x0,x0] = 1
        B[xf,xf] = 1
    elif version == 'x0wb':
        B = np.zeros((num_parcels,num_parcels))
        B[np.eye(num_parcels) == 1] = 5*10e-5
        B[x0,x0] = 1
    elif version == 'xfwb':
        B = np.zeros((num_parcels,num_parcels))
        B[np.eye(num_parcels) == 1] = 5*10e-5
        B[xf,xf] = 1

    return B


def subsample_state(x, subsample_size):
    x_tmp = np.zeros(x.size).astype(bool)
    
    sample = np.random.choice(np.where(x == True)[0], size = subsample_size, replace = False)
    
    x_tmp[sample] = True

    return x_tmp


def get_time_vec(T, n_taylor):
    return np.power(T, np.arange(0,n_taylor))


def minimum_energy_taylor(A, T, B, x0, xf, c = 1, n_taylor = 10, drop_taylor = 0):
    num_parcels = A.shape[0] # Number of nodes

    # Normalize and eigendecompose
    u, s, vt = np.linalg.svd(A)                 # singluar value decomposition
    A = A/(c + s[0]) - np.eye(A.shape[0])       # Matrix normalization
    w, v = np.linalg.eig(A)                     # Eigenvalue decomposition

    # Define eigenvalue powers (~.25)
    w = np.reshape(w,(num_parcels,1))
    p = np.reshape(np.arange(0,n_taylor),(1,n_taylor))
    W = np.power(w,p)

    # Define Taylor series coefficients
    tc = np.zeros(n_taylor)
    for i in np.arange(0,n_taylor):
        tc[i] = 1/math.factorial(i)
    if drop_taylor > 0:
        tc[drop_taylor] = 0
    tc = np.reshape(tc,(1,n_taylor))

    # Multiple eigenvalues with coefficients
    W = np.multiply(W,tc)

    # Define time matrix
    nN = 1000
    t_mat = np.zeros((1,n_taylor,nN))
    for i in np.arange(0,nN):
        t_mat[0,:,i] = get_time_vec((T/nN)*i,n_taylor)

    # Perform numerical integration (~.25)
    WT = np.dot(W,t_mat)
    WT = np.multiply(WT,np.reshape(WT,(1,num_parcels,nN)))

    # Define Gramian
    P = np.matmul(v.T,B)
    P = np.matmul(P,P.T)
    P = np.reshape(P,(num_parcels,num_parcels,1))
    WcM = np.multiply(WT,P)
    Wc = np.matmul(np.matmul(v,np.sum(WcM,2)),v.T) * (T/nN)

    # State transition
    WPr = np.multiply(W, np.reshape(get_time_vec(T,n_taylor),(1,n_taylor)))
    EV = np.matmul(v,np.matmul(np.diag(np.sum(WPr,1)),v.T))
    Phi = np.dot(EV, x0) - xf
        
    E = np.dot(Phi.T,np.linalg.solve(Wc,Phi))
    
    return E

# --------------------------------------------------------------------------------------------------------------------
# load data
A = np.load(A_file)
num_parcels = A.shape[0]

drop_taylor_matrix = np.load(drop_taylor_file)

gradients = np.loadtxt(gradients_file)
n_clusters = drop_taylor_matrix.shape[0]
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(gradients)

# cluster 2D gradient space
unique, counts = np.unique(kmeans.labels_, return_counts = True)
subsample_size = np.min(counts)
print(subsample_size)

# --------------------------------------------------------------------------------------------------------------------
E = np.zeros((n_clusters, n_clusters, n_subsamples))
Eto = np.zeros((n_clusters, n_clusters, n_subsamples))

for i in np.arange(n_clusters):
    for j in np.arange(n_clusters):
        if i != j:
            np.random.seed(0)

            x0 = kmeans.labels_ == i
            xf = kmeans.labels_ == j

            for k in np.arange(n_subsamples):
                x0_tmp = subsample_state(x0, subsample_size)
                xf_tmp = subsample_state(xf, subsample_size)

                B = get_B_matrix(x0_tmp, xf_tmp, version = B_ver)

                E[i,j,k] = minimum_energy_taylor(A, T, B, x0_tmp, xf_tmp, n_taylor = n_taylor, drop_taylor = 0)
                if drop_taylor_matrix[i,j] == 0:
                    Eto[i,j,k] = E[i,j,k].copy()
                else:
                    Eto[i,j,k] = minimum_energy_taylor(A, T, B, x0_tmp, xf_tmp, n_taylor = n_taylor, drop_taylor = drop_taylor_matrix[i,j])

np.save(os.path.join(outputdir,subjid+'_T-'+str(T)+'_B-'+B_ver+'-g'+str(n_clusters)+'_E'), E)
np.save(os.path.join(outputdir,subjid+'_T-'+str(T)+'_B-'+B_ver+'-g'+str(n_clusters)+'_Eto'), Eto)

print('Finished!')
