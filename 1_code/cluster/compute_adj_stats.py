import argparse

import os, sys
import numpy as np
from scipy.linalg import svd
import scipy as sp
import numpy.matlib

from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from bct.utils import weight_conversion
from bct.algorithms.distance import distance_wei, distance_wei_floyd, retrieve_shortest_path
from bct.algorithms.reference import randmio_und

# --------------------------------------------------------------------------------------------------------------------
# parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument("-subjid", help="label for participant", dest="subjid", default=None, type=str)
parser.add_argument("-A_file", help="path and file to adjacency matrix", dest="A_file", default=None, type=str)
parser.add_argument("-gradients_file", help="", dest="gradients_file", default=None, type=str)
parser.add_argument("-outputdir", help="output directory", dest="outputdir", default=None, type=str)

args = parser.parse_args()

subjid = args.subjid
A_file = args.A_file
gradients_file = args.gradients_file
outputdir = args.outputdir

# --------------------------------------------------------------------------------------------------------------------
# functions
def get_gradient_variance(shortest_path, gradients, return_abs = False):

    # calculate the differences between coordinates in adjacent nodes along shortest path
    gradient_diff = np.diff(gradients[shortest_path,:], axis = 0)

    if return_abs == True:
         gradient_diff = np.abs(gradient_diff)

    mean_diff = np.mean(gradient_diff, axis = 0) # mean the differences along the shortest path
    var_diff = np.var(gradient_diff, axis = 0) # get the variance of the differences along the shortest path

    euclidean_var = np.var(np.sqrt(np.sum(np.square(gradient_diff), axis = 1))) # get the variance of the euclidean distance
    
    return mean_diff, var_diff, euclidean_var


def get_adj_stats(A, gradients, return_abs = False):

    num_parcels = A.shape[0]
    
    # convert to distance matrix
    D, hops, Pmat = distance_wei_floyd(A, transform = 'inv')
    
    # get transmodal and sensorimotor-visual traversal variance
    tm_con = np.zeros((num_parcels,num_parcels)); tm_con[:] = np.nan
    tm_var = np.zeros((num_parcels,num_parcels)); tm_var[:] = np.nan
    smv_con = np.zeros((num_parcels,num_parcels)); smv_con[:] = np.nan
    smv_var = np.zeros((num_parcels,num_parcels)); smv_var[:] = np.nan
    joint_var = np.zeros((num_parcels,num_parcels)); joint_var[:] = np.nan

    for i in np.arange(num_parcels):
        for j in np.arange(num_parcels):
            shortest_path = retrieve_shortest_path(i,j,hops,Pmat)
            if len(shortest_path) != 0:
                shortest_path = shortest_path.flatten()
                mean_diff, var_diff, euclidean_var = get_gradient_variance(shortest_path, gradients, return_abs = return_abs)

                tm_con[i,j] = mean_diff[0]
                tm_var[i,j] = var_diff[0]
                smv_con[i,j] = mean_diff[1]
                smv_var[i,j] = var_diff[1]
                joint_var[i,j] = euclidean_var

    return D, hops, Pmat, tm_con, tm_var, smv_con, smv_var, joint_var

# --------------------------------------------------------------------------------------------------------------------
# load data
A = np.load(A_file)
gradients = np.loadtxt(gradients_file)

# --------------------------------------------------------------------------------------------------------------------
adj_stats = {}

D, hops, Pmat, tm_con, tm_var, smv_con, smv_var, joint_var = get_adj_stats(A, gradients, return_abs = False)
adj_stats['D'] = D
adj_stats['hops'] = hops
adj_stats['Pmat'] = Pmat
adj_stats['tm_con'] = tm_con
adj_stats['tm_var'] = tm_var
adj_stats['smv_con'] = smv_con
adj_stats['smv_var'] = smv_var
adj_stats['joint_var'] = joint_var

D, hops, Pmat, tm_con, tm_var, smv_con, smv_var, joint_var = get_adj_stats(A, gradients, return_abs = True)
adj_stats['tm_con_abs'] = tm_con
adj_stats['tm_var_abs'] = tm_var
adj_stats['smv_con_abs'] = smv_con
adj_stats['smv_var_abs'] = smv_var

np.save(os.path.join(outputdir,subjid+'_adj_stats'), adj_stats)

print('Finished!')
