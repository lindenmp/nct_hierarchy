import argparse

import os
import numpy as np
from scipy.linalg import svd
import scipy as sp
import numpy.matlib
import math
from sklearn.cluster import KMeans
from bct.algorithms.distance import retrieve_shortest_path, distance_wei_floyd

# --------------------------------------------------------------------------------------------------------------------
# parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument("-subjid", help="label for participant", dest="subjid", default=None, type=str)
parser.add_argument("-A_file", help="path and file to adjacency matrix", dest="A_file", default=None, type=str)
parser.add_argument("-T", help="", dest="T", default=None, type=int)
parser.add_argument("-control", help="", dest="control", default=None, type=str)
parser.add_argument("-gradients_file", help="", dest="gradients_file", default=None, type=str)
parser.add_argument("-n_clusters", help="", dest="n_clusters", default=None, type=int)
parser.add_argument("-outputdir", help="output directory", dest="outputdir", default=None, type=str)
parser.add_argument("-i", help="", dest="i", default=None, type=int)
parser.add_argument("-j", help="", dest="j", default=None, type=int)

args = parser.parse_args()

subjid = args.subjid
A_file = args.A_file
T = args.T
control = args.control
gradients_file = args.gradients_file
n_clusters = args.n_clusters
outputdir = args.outputdir
i = args.i
j = args.j

# --------------------------------------------------------------------------------------------------------------------
# functions
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


# --------------------------------------------------------------------------------------------------------------------
# outputdir
# if not os.path.exists(outputdir): os.makedirs(outputdir)

# load data
A = np.load(A_file)
num_parcels = A.shape[0]

gradients = np.loadtxt(gradients_file)
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(gradients)

unique, counts = np.unique(kmeans.labels_, return_counts = True)
subsample_size = np.min(counts)
print(subsample_size)

D, hops, Pmat = distance_wei_floyd(A, transform = 'inv')
# num_taylor = np.max(hops).astype(int)+1
num_taylor = np.floor(np.mean(hops)).astype(int)+1
print(num_taylor)

# # --------------------------------------------------------------------------------------------------------------------
# E = np.zeros((n_clusters, n_clusters, num_taylor))

# num_subsamples = 5

# for i in np.arange(n_clusters):
#     for j in np.arange(n_clusters):
        
#         x0 = kmeans.labels_ == i
#         xf = kmeans.labels_ == j

#         E_tmp = np.zeros((num_subsamples,num_taylor))
        
#         np.random.seed(0)
#         for k in np.arange(num_subsamples):
#             x0_tmp = subsample_state(x0, subsample_size)
#             xf_tmp = subsample_state(xf, subsample_size)

#             B = get_B_matrix(x0_tmp, xf_tmp, control = control)

#             for t in np.arange(0,num_taylor):
#                 E_tmp[k,t] = minimum_energy_taylor(A,T,B,x0_tmp,xf_tmp, num_taylor = num_taylor, drop_taylor = t)

#         E_tmp = np.nanmean(E_tmp, axis = 0)    

#         E[i,j,:] = E_tmp

# np.save(os.path.join(outputdir,subjid+'_control-grad'+str(n_clusters)+'_B-'+control+'_T-'+str(T)+'_E'), E)

# --------------------------------------------------------------------------------------------------------------------
# i=0
# j=1

x0 = kmeans.labels_ == i
xf = kmeans.labels_ == j

num_subsamples = 50
E_tmp = np.zeros((num_subsamples,num_taylor))

np.random.seed(0)
for k in np.arange(num_subsamples):
    x0_tmp = subsample_state(x0, subsample_size)
    xf_tmp = subsample_state(xf, subsample_size)

    B = get_B_matrix(x0_tmp, xf_tmp, control = control)

    for t in np.arange(0,num_taylor):
        E_tmp[k,t] = minimum_energy_taylor(A,T,B,x0_tmp,xf_tmp, num_taylor = num_taylor, drop_taylor = t)

E = np.nanmean(E_tmp, axis = 0)    

np.save(os.path.join(outputdir,subjid+'_control-grad'+str(n_clusters)+'_B-'+control+'_T-'+str(T)+'_E_'+str(i)+str(j)), E)

print('Finished!')
