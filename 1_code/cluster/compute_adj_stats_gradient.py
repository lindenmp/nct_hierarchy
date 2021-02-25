import argparse

import os
import numpy as np
from scipy.linalg import svd
import scipy as sp
import numpy.matlib

from sklearn.cluster import KMeans
from bct.utils import weight_conversion
from bct.algorithms.distance import distance_wei, distance_wei_floyd, retrieve_shortest_path
from bct.algorithms.reference import randmio_und

# --------------------------------------------------------------------------------------------------------------------
# parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument("-subjid", help="label for participant", dest="subjid", default=None, type=str)
parser.add_argument("-A_file", help="path and file to adjacency matrix", dest="A_file", default=None, type=str)
parser.add_argument("-gradients_file", help="", dest="gradients_file", default=None, type=str)
parser.add_argument("-n_clusters", help="", dest="n_clusters", default=None, type=int)
parser.add_argument("-outputdir", help="output directory", dest="outputdir", default=None, type=str)
parser.add_argument("-surr_seed", help="", dest="surr_seed", default=-1, type=int)

args = parser.parse_args()

subjid = args.subjid
A_file = args.A_file
gradients_file = args.gradients_file
n_clusters = args.n_clusters
outputdir = args.outputdir
surr_seed = args.surr_seed

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


def get_gradient_variance(shortest_path, gradients, return_abs = False):

    # calculate the differences between coordinates in adjacent nodes along shortest path
    gradient_diff = np.diff(gradients[shortest_path,:], axis = 0)

    if return_abs == True:
         gradient_diff = np.abs(gradient_diff)

    mean_diff = np.mean(gradient_diff, axis = 0) # mean the differences along the shortest path
    var_diff = np.var(gradient_diff, axis = 0) # get the variance of the differences along the shortest path

    euclidean_var = np.var(np.sqrt(np.sum(np.square(gradient_diff), axis = 1))) # get the variance of the euclidean distance
    
    return mean_diff, var_diff, euclidean_var


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

    # downsample transmodal convergence to cluster-based states
    tm_con = matrix_to_states(tm_tmp, cluster_labels)
    tm_var = matrix_to_states(tm_var_tmp, cluster_labels)

    smv_con = matrix_to_states(smv_tmp, cluster_labels)
    smv_var = matrix_to_states(smv_var_tmp, cluster_labels)
    
    joint_var = matrix_to_states(joint_var_tmp, cluster_labels)
            
    return D_mean, hops_mean, tm_con, tm_var, smv_con, smv_var, joint_var


# --------------------------------------------------------------------------------------------------------------------
# outputdir
# if not os.path.exists(outputdir): os.makedirs(outputdir)

# load data
A = np.load(A_file)

gradients = np.loadtxt(gradients_file)
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(gradients)

# --------------------------------------------------------------------------------------------------------------------
if surr_seed != -1:
    num_edge_swaps = int(5*10e4)

    num_parcels = A.shape[0]
    num_connections = num_parcels*num_parcels-num_parcels

    num_iter = int(num_edge_swaps/num_connections)

    np.random.seed(surr_seed)
    A, eff = randmio_und(A, itr = num_iter)

adj_stats = {}

D_mean, hops_mean, tm_con, tm_var, smv_con, smv_var, joint_var = get_adj_stats(A, gradients, kmeans.labels_, return_abs = False)
adj_stats['D_mean'] = D_mean
adj_stats['hops_mean'] = hops_mean
adj_stats['tm_con'] = tm_con
adj_stats['tm_var'] = tm_var
adj_stats['smv_con'] = smv_con
adj_stats['smv_var'] = smv_var
adj_stats['joint_var'] = joint_var

D_mean, hops_mean, tm_con, tm_var, smv_con, smv_var, joint_var = get_adj_stats(A, gradients, kmeans.labels_, return_abs = True)
adj_stats['tm_con_abs'] = tm_con
adj_stats['tm_var_abs'] = tm_var
adj_stats['smv_con_abs'] = smv_con
adj_stats['smv_var_abs'] = smv_var

if surr_seed != -1:
    # np.save(os.path.join(outputdir,subjid+'_surr'+str(surr_seed)), A)
    np.save(os.path.join(outputdir,subjid+'_surr'+str(surr_seed)+'_grad'+str(n_clusters)+'_adj_stats'), adj_stats)
else:
    np.save(os.path.join(outputdir,subjid+'_grad'+str(n_clusters)+'_adj_stats'), adj_stats)


print('Finished!')
