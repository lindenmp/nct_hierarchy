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

parser.add_argument("-control", help="", dest="control", default='minimum', type=str)
parser.add_argument("-T", help="", dest="T", default=1, type=int)
parser.add_argument("-B_ver", help="", dest="B_ver", default='x0xfwb', type=str)
parser.add_argument("-rho", help="", dest="rho", default=1, type=float)
parser.add_argument("-n_subsamples", help="number of times a state is subsampled", dest="n_subsamples", default=50, type=int)

parser.add_argument("-gradients_file", help="", dest="gradients_file", default=None, type=str)
parser.add_argument("-n_clusters", help="", dest="n_clusters", default=20, type=int)
parser.add_argument("-i", help="", dest="i", default=-1, type=int)
parser.add_argument("-j", help="", dest="j", default=-1, type=int)

args = parser.parse_args()

subjid = args.subjid
A_file = args.A_file
outputdir = args.outputdir

control = args.control
T = args.T
B_ver = args.B_ver
rho = args.rho
n_subsamples = args.n_subsamples

gradients_file = args.gradients_file
n_clusters = args.n_clusters
i = args.i
j = args.j

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
print(control)

# load data
A = np.load(A_file)
num_parcels = A.shape[0]
S = np.eye(num_parcels)

gradients = np.loadtxt(gradients_file)
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(gradients)

# cluster 2D gradient space
unique, counts = np.unique(kmeans.labels_, return_counts = True)
subsample_size = np.min(counts)
print(subsample_size)

if control == 'minimum_taylor':
    # find number of hops in matrix
    D, hops, Pmat = distance_wei_floyd(A, transform = 'inv')

    # use hops matrix to determine taylor polynomial order
    # n_taylor = np.max(hops).astype(int)+1
    n_taylor = np.floor(np.mean(hops)).astype(int)+1
    print(n_taylor)

# --------------------------------------------------------------------------------------------------------------------
if i == -1 and j == -1:
    print('looping i,j...')

    if control == 'minimum_taylor':
        E = np.zeros((n_clusters, n_clusters, n_subsamples, n_taylor))
    else:
        E = np.zeros((n_clusters, n_clusters, n_subsamples))
        n_err = np.zeros((n_clusters, n_clusters, n_subsamples))

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

                    if control == 'minimum':
                        x, u, n_err[i,j,k] = minimum_energy(A, T, B, x0_tmp, xf_tmp)
                        u = np.multiply(np.matlib.repmat(B[np.eye(num_parcels) == 1], u.shape[0],1), u) # scale energy
                        E[i,j,k] = np.sum(np.square(u))

                    elif control == 'minimum_nonh':
                        E[i,j,k] = minimum_energy_nonh(A, T, B, x0_tmp, xf_tmp)

                    elif control == 'minimum_taylor':
                        for t in np.arange(0,n_taylor):
                            E[i,j,k,t] = minimum_energy_taylor(A, T, B, x0_tmp, xf_tmp, n_taylor = n_taylor, drop_taylor = t)

                    elif control == 'optimal':
                        x, u, n_err[i,j,k] = optimal_energy(A, T, B, x0_tmp, xf_tmp, rho, S) # get optimal control energy
                        u = np.multiply(np.matlib.repmat(B[np.eye(num_parcels) == 1], u.shape[0],1), u) # scale energy
                        E[i,j,k] = np.sum(np.square(u))


    if control == 'minimum':
        np.save(os.path.join(outputdir,subjid+'_'+control+'_T-'+str(T)+'_B-'+B_ver+'-g'+str(n_clusters)+'_E'), E)
        np.save(os.path.join(outputdir,subjid+'_'+control+'_T-'+str(T)+'_B-'+B_ver+'-g'+str(n_clusters)+'_n_err'), n_err)
    elif control == 'minimum_taylor' or control == 'minimum_nonh':
        np.save(os.path.join(outputdir,subjid+'_'+control+'_T-'+str(T)+'_B-'+B_ver+'-g'+str(n_clusters)+'_E'), E)
    elif control == 'optimal':
        np.save(os.path.join(outputdir,subjid+'_'+control+'_T-'+str(T)+'_B-'+B_ver+'_rho-'+str(rho)+'-g'+str(n_clusters)+'_E'), E)
        np.save(os.path.join(outputdir,subjid+'_'+control+'_T-'+str(T)+'_B-'+B_ver+'_rho-'+str(rho)+'-g'+str(n_clusters)+'_n_err'), n_err)
else:
    print('reading i,j from function inputs...')

    if control == 'minimum_taylor':
        E = np.zeros((n_subsamples, n_taylor))
    else:
        E = np.zeros(n_subsamples)
        n_err = np.zeros(n_subsamples)

    x0 = kmeans.labels_ == i
    xf = kmeans.labels_ == j
    
    np.random.seed(0)
    for k in np.arange(n_subsamples):
        x0_tmp = subsample_state(x0, subsample_size)
        xf_tmp = subsample_state(xf, subsample_size)

        B = get_B_matrix(x0_tmp, xf_tmp, version = B_ver)

        if control == 'minimum':
            x, u, n_err[k] = minimum_energy(A, T, B, x0_tmp, xf_tmp)
            u = np.multiply(np.matlib.repmat(B[np.eye(num_parcels) == 1], u.shape[0],1), u) # scale energy
            E[k] = np.sum(np.square(u))

        elif control == 'minimum_nonh':
            E[k] = minimum_energy_nonh(A, T, B, x0_tmp, xf_tmp)

        elif control == 'minimum_taylor':
            for t in np.arange(0,n_taylor):
                E[k,t] = minimum_energy_taylor(A, T, B, x0_tmp, xf_tmp, n_taylor = n_taylor, drop_taylor = t)

        elif control == 'optimal':
            x, u, n_err[k] = optimal_energy(A, T, B, x0_tmp, xf_tmp, rho, S) # get optimal control energy
            u = np.multiply(np.matlib.repmat(B[np.eye(num_parcels) == 1], u.shape[0],1), u) # scale energy
            E[k] = np.sum(np.square(u))


    if control == 'minimum':
        np.save(os.path.join(outputdir,subjid+'_'+control+'_T-'+str(T)+'_B-'+B_ver+'-g'+str(n_clusters)+'_E_i'+str(i)+'j'+str(j)), E)
        np.save(os.path.join(outputdir,subjid+'_'+control+'_T-'+str(T)+'_B-'+B_ver+'-g'+str(n_clusters)+'_n_err_i'+str(i)+'j'+str(j)), n_err)
    elif control == 'minimum_taylor' or control == 'minimum_nonh':
        np.save(os.path.join(outputdir,subjid+'_'+control+'_T-'+str(T)+'_B-'+B_ver+'-g'+str(n_clusters)+'_E_i'+str(i)+'j'+str(j)), E)
    elif control == 'optimal':
        np.save(os.path.join(outputdir,subjid+'_'+control+'_T-'+str(T)+'_B-'+B_ver+'_rho-'+str(rho)+'-g'+str(n_clusters)+'_E_i'+str(i)+'j'+str(j)), E)
        np.save(os.path.join(outputdir,subjid+'_'+control+'_T-'+str(T)+'_B-'+B_ver+'_rho-'+str(rho)+'-g'+str(n_clusters)+'_n_err_i'+str(i)+'j'+str(j)), n_err)


print('Finished!')
