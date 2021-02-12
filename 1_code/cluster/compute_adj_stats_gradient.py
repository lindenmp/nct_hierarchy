import argparse

import os
import numpy as np
from scipy.linalg import svd
import scipy as sp
import numpy.matlib

from sklearn.cluster import KMeans
from bct.utils import weight_conversion
from bct.algorithms.distance import distance_wei, distance_wei_floyd, retrieve_shortest_path

# --------------------------------------------------------------------------------------------------------------------
# parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument("-subjid", help="label for participant", dest="subjid", default=None, type=str)
parser.add_argument("-A_file", help="path and file to adjacency matrix", dest="A_file", default=None, type=str)
parser.add_argument("-gradients_file", help="", dest="gradients_file", default=None, type=str)
parser.add_argument("-n_clusters", help="", dest="n_clusters", default=None, type=int)
parser.add_argument("-outputdir", help="output directory", dest="outputdir", default=None, type=str)

args = parser.parse_args()

subjid = args.subjid
A_file = args.A_file
gradients_file = args.gradients_file
n_clusters = args.n_clusters
outputdir = args.outputdir

# --------------------------------------------------------------------------------------------------------------------
# functions
def matrix_to_states(x, cluster_labels):
    
    unique, counts = np.unique(cluster_labels, return_counts = True)
    n_clusters = len(unique)
    
    x_out = np.zeros((n_clusters,n_clusters))
    for i in np.arange(n_clusters):
        for j in np.arange(n_clusters):
            x_out[i,j] = x[cluster_labels == i,:].mean(axis = 0)[cluster_labels == j].mean()
            
    return x_out


def get_tm_convergence(shortest_path, gradients, return_abs = False):

    tm_convergence = np.diff(gradients[shortest_path,:], axis = 0)

    if return_abs == True:
         tm_convergence = np.abs(tm_convergence)
    
    return np.mean(tm_convergence, axis = 0), np.var(tm_convergence, axis = 0)


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

    for i in np.arange(num_parcels):
        for j in np.arange(num_parcels):
            shortest_path = retrieve_shortest_path(i,j,hops,Pmat)
            if len(shortest_path) != 0:
                shortest_path = shortest_path.flatten()
                x_mean, x_var = get_tm_convergence(shortest_path, gradients, return_abs = return_abs)
                tm_tmp[i,j] = x_mean[0]
                tm_var_tmp[i,j] = x_var[0]
                smv_tmp[i,j] = x_mean[1]
                smv_var_tmp[i,j] = x_var[1]

    # downsample transmodal convergence to cluster-based states
    tm_con = matrix_to_states(tm_tmp, cluster_labels)
    tm_var = matrix_to_states(tm_var_tmp, cluster_labels)
    smv_con = matrix_to_states(smv_tmp, cluster_labels)
    smv_var = matrix_to_states(smv_var_tmp, cluster_labels)
            
    return D_mean, hops_mean, tm_con, tm_var, smv_con, smv_var


# --------------------------------------------------------------------------------------------------------------------
# outputdir
if not os.path.exists(outputdir): os.makedirs(outputdir)

# load data
A = np.load(A_file)

gradients = np.loadtxt(gradients_file)
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(gradients)

# --------------------------------------------------------------------------------------------------------------------
adj_stats = {}

D_mean, hops_mean, tm_con, tm_var, smv_con, smv_var = get_adj_stats(A, gradients, kmeans.labels_, return_abs = False)
adj_stats['D_mean'] = D_mean
adj_stats['hops_mean'] = hops_mean
adj_stats['tm_con'] = tm_con
adj_stats['tm_var'] = tm_var
adj_stats['smv_con'] = smv_con
adj_stats['smv_var'] = smv_var

D_mean, hops_mean, tm_con, tm_var, smv_con, smv_var = get_adj_stats(A, gradients, kmeans.labels_, return_abs = True)
adj_stats['tm_con_abs'] = tm_con
adj_stats['tm_var_abs'] = tm_var
adj_stats['smv_con_abs'] = smv_con
adj_stats['smv_var_abs'] = smv_var

np.save(os.path.join(outputdir,subjid+'_grad'+str(n_clusters)+'_adj_stats'), adj_stats)

print('Finished!')
