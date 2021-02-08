import argparse

import os
import numpy as np
from scipy.linalg import svd
import scipy as sp
import numpy.matlib

# --------------------------------------------------------------------------------------------------------------------
# parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument("-subjid", help="label for participant", dest="subjid", default=None, type=str)
parser.add_argument("-A_file", help="path and file to adjacency matrix", dest="A_file", default=None, type=str)
parser.add_argument("-T", help="", dest="T", default=None, type=float)
parser.add_argument("-rho", help="", dest="rho", default=None, type=float)
parser.add_argument("-control", help="", dest="control", default=None, type=str)
parser.add_argument("-outputdir", help="output directory", dest="outputdir", default=None, type=str)

args = parser.parse_args()

subjid = args.subjid
A_file = args.A_file
T = args.T
rho = args.rho
control = args.control
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

    if type(x0[0]) == numpy.bool_:
        x0 = x0.astype(float)
    if x0.ndim == 1:
        x0 = x0.reshape(-1,1)

    if type(xf[0]) == numpy.bool_:
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

# --------------------------------------------------------------------------------------------------------------------
# outputdir
if not os.path.exists(outputdir): os.makedirs(outputdir)

# load data
A = np.load(A_file)
num_parcels = A.shape[0]

S = np.eye(num_parcels)

# --------------------------------------------------------------------------------------------------------------------
# Single region control
E = np.zeros((num_parcels,num_parcels))
n_err = np.zeros((num_parcels,num_parcels))

for i in np.arange(num_parcels):
    for j in np.arange(num_parcels):

        x0 = np.zeros(num_parcels).astype(bool)
        x0[i] = 1

        xf = np.zeros(num_parcels).astype(bool)
        xf[j] = 1

        B = get_B_matrix(x0, xf, control = control)
 
        x, u, n_err[i,j] = optimal_energy(A,T,B,x0,xf,rho,S) # get optimal control energy
        u = np.multiply(np.matlib.repmat(B[np.eye(num_parcels) == 1],u.shape[0],1),u) # scale energy
        E[i,j] = np.sum(np.square(u)) # integrate

np.save(os.path.join(outputdir,subjid+'_control-sr_B-'+control+'_T-'+str(T)+'_rho-'+str(rho)+'_E'), E)
np.save(os.path.join(outputdir,subjid+'_control-st_B-'+control+'_T-'+str(T)+'_rho-'+str(rho)+'_n_err'), n_err)


print('Finished!')
