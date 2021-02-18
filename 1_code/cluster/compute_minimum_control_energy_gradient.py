import argparse

import os
import numpy as np
from scipy.linalg import svd
import scipy as sp
import numpy.matlib

from sklearn.cluster import KMeans

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

args = parser.parse_args()

subjid = args.subjid
A_file = args.A_file
T = args.T
control = args.control
gradients_file = args.gradients_file
n_clusters = args.n_clusters
outputdir = args.outputdir

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


# --------------------------------------------------------------------------------------------------------------------
# outputdir
if not os.path.exists(outputdir): os.makedirs(outputdir)

# load data
A = np.load(A_file)
num_parcels = A.shape[0]

gradients = np.loadtxt(gradients_file)
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(gradients)

unique, counts = np.unique(kmeans.labels_, return_counts = True)
subsample_size = np.min(counts)
print(subsample_size)

# --------------------------------------------------------------------------------------------------------------------
E = np.zeros((n_clusters, n_clusters))
n_err = np.zeros((n_clusters, n_clusters))

num_subsamples = 50

for i in np.arange(n_clusters):
    for j in np.arange(n_clusters):
        
        x0 = kmeans.labels_ == i
        xf = kmeans.labels_ == j

        E_tmp = np.zeros(num_subsamples)
        n_err_tmp = np.zeros(num_subsamples)
        
        np.random.seed(0)
        for k in np.arange(num_subsamples):
            x0_tmp = subsample_state(x0, subsample_size)
            xf_tmp = subsample_state(xf, subsample_size)

            B = get_B_matrix(x0_tmp, xf_tmp, control = control)

            x, u, n_err_tmp[k] = minimum_energy(A,T,B,x0_tmp,xf_tmp)
            u = np.multiply(np.matlib.repmat(B[np.eye(num_parcels) == 1],u.shape[0],1),u) # scale energy
            E_tmp[k] = np.sum(np.square(u))

        n_err[i,j] = np.nanmean(n_err_tmp)
        E[i,j] = np.nanmean(E_tmp)

np.save(os.path.join(outputdir,subjid+'_control-grad'+str(n_clusters)+'_B-'+control+'_T-'+str(T)+'_E'), E)
np.save(os.path.join(outputdir,subjid+'_control-grad'+str(n_clusters)+'_B-'+control+'_T-'+str(T)+'_n_err'), n_err)

print('Finished!')
