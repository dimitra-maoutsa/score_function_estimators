# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 14:15:09 2022

@author: maout
"""

import math
import numpy as np

from functools import reduce
from scipy.spatial.distance import cdist
import numba
from kernels.RBF_kernel import RBF
##this is the masked Hyvarinen version DO NOT USE!!!
def score_function_multid_bounded_all_dims(X,Z,func_out=False, C=0.001,
                                             kern ='RBF',l=None, domain=None,
                                             opt_hyp=False,
                                             method='KF'):
    """
    Sparse kernel based estimation of multidimensional logarithmic gradient 
    of empirical density defined on a bounded support represented 
    by samples X for all dimensions simultaneously. 
    This is the bounded estimator with the weighting function like Hyvarinen.
    
    - When `funct_out == False`: computes grad-log at the sample points.
    - When `funct_out == True`: return a function for the grad log to be employed for interpolation/estimation of grad log 
                               in the vicinity of the samples.
                               
    Uses kernel classes.
    
    Parameters
    -----------
            X: N x dim array,
               N samples from the density (N x dim), where dim>=2 the 
               dimensionality of the system.
            Z: M x dim array,
              inducing points points (M x dim).
            func_out : Boolean, 
                      True returns function, 
                      if False returns grad-log-p evaluated on samples X.                    
            l: float or array-like,
               lengthscale of rbf kernel (scalar or vector of size dim).
            C: float,
              weighting constant 
              (leave it at default value to avoid unreasonable contraction 
              of deterministic trajectories).
            domain: 2 x dim ndarray or arraylike,
                    min and max bounds of the domain over which the 
                    density is defined 
                    ( [[min_dim1, max_dim1],[min_dim2, max_dim2]] )
            kern: string,
                options:
                    - 'RBF': radial basis function/Gaussian kernel  
                    - 'periodic': periodic, not functional yet.   
            opt_hyp: boolean,
                indicates whether hyperparameter optimisation will be pursued
            method: string,
                method for hyperparameter optimisation
                    options:
                        - 'LOO': leave one out estimator
                        - 'KF' : kernel flow
            
    Returns
    -------
        res1: array with logarithmic gradient of the density  N_s x dim or function 
                 that accepts as inputs 2dimensional arrays of dimension (K x dim), where K>=1.
    
    """
    
    if kern=='RBF':
        ###initialise kernel objects
        ker_xz = RBF()
        ker_zz = RBF()
    elif kern=='periodic':
        print('not implemented yet!')
        
    if domain is None:
        print('Please give the domain of support of the sampled density!')
        print('If the domain is unbounded please use the default score estimator!')
        

    N, dim = X.shape
    
    if (l is None) and (opt_hyp == False):
        ## if l is not provided, and no optimisation is wanted
        ## the engthscle will be the default option of the std
        ker_xz.set_multil = True
        ker_zz.set_multil = True
        
        l = 2 * np.std(X, axis=0) 
        ker_xz.set_lnthsc(l)
        ker_zz.set_lnthsc(l)
        
    elif (l is None) and (opt_hyp == True):
        ## if l is not provided and optimisation  is required
        ## set the indicator in the kernel
        ker_xz.set_multil = True
        ker_zz.set_multil = True
        
        l = ker_xz.set_optimise_hyperparams(True, method)
        ker_xz.set_lnthsc(l)
        ker_zz.set_lnthsc(l)

    elif isinstance(l, (list, tuple, np.ndarray)):
        ## if l is given and is a list-like object
        multil = True
        ##set lengthscale
        ker_xz.set_multil = True
        ker_zz.set_multil = True
        ker_xz.set_lnthsc(l)
        ker_zz.set_lnthsc(l)
        
    else:
        ## if l is given and is a scalar, i.e. same l for all dimensions
        ## this option should be avoided for most settings
        ## but it is provided here for completeness
        print('only one fixed lenghtscale')
        ker_xz.set_multil = True
        ker_zz.set_multil = True
        ker_xz.set_lnthsc(l)
        ker_zz.set_lnthsc(l)
        
    ###weight function - here I use the identity 
    Gs = np.zeros((N, N, dim))
    for di in range(dim):
        if (domain[di][0] is None) and (domain[di][1] is None):
            ### if no bounds then Gs = x - this should serve as a sanity check 
            ### but also for the dimensions without bounds
            Gs[:,:,di] = np.power(np.diag(X[:,di]),2)
            
        elif (domain[di][0] is not None) and (domain[di][1] is None):
            ### g(x) = x- a
            Gs[:,:,di] = np.power(np.diag(X[:,di] - domain[di][0]),2)
        elif (domain[di][0] is None) and (domain[di][1] is not None):
            ### g(x) = b - x
            Gs[:,:,di] = np.power(np.diag(domain[di][1] - X[:,di]),2)
        elif (domain[di][0] is not None) and (domain[di][1] is not None):
            ### g(x) = min( a-x, b - x)
            Gs[:,:,di] = np.power(np.diag( np.minimum(X[:,di] - domain[di][0],
                                             domain[di][1] - X[:,di]) ),2)
        
     
    
        
    K_xz = ker_xz.K(X,Z) 
    Ks = ker_zz.K(Z,Z)    
    gradx_K = ker_xz.grdx_K_all()   ##shape: N x M x dim
    
       
    Ksinv = np.linalg.inv(Ks+ 1e-3 * np.eye(Z.shape[0]))
    
    
    if func_out==False:
        res1 = np.zeros((ker_xz.N, ker_xz.dim)) 
    
    for di in range(dim):
        A = K_xz.T @ Gs[:,:,di] @ K_xz      
        ###just for future developments here  DG stands for the derivative of 
        ###function G
        DG = np.diag( X[:,di])
        sumgradG_K = np.sum( DG@ K_xz , axis=0 )  #N x M --> after sum: 1 x M
    
        print(Gs[:,:,di].shape)
        print(gradx_K.shape)
        
        print( ( np.einsum( 'ii,ijk->ijk'   ,Gs[:,:,di] ,gradx_K ).shape ))
        
        
        
        sumgradx_K =  np.sum( np.einsum( 'ii,ijk->ijk',Gs[:,:,di] ,gradx_K ), axis=0) ##last axis will have the gradient for each dimension ### shape (M, dim)
    
        if func_out==False: #if output wanted is evaluation at data points            
                        
            ### evaluatiion at data points
                                                                                                                                ##sumgradG_K in more general case should be added per dimension
            res1[:,di] =  - K_xz @ np.linalg.inv( C*np.eye(Z.shape[0], Z.shape[0])+ 1e-6 * np.eye(Z.shape[0]) + Ksinv @ A ) @ Ksinv  @ (sumgradx_K[:,di] +sumgradG_K  )
            
        else:           
            #### for function output 
            if multil:      
                if kern=='RBF':      
                    K_sz = lambda x: reduce(np.multiply, [ np.exp(-cdist(x[:,iii].reshape(-1,1), Z[:,iii].reshape(-1,1),'sqeuclidean')/(2*l[iii]*l[iii])) for iii in range(x.shape[1]) ])        
                    
                elif kern=='periodic':
                    K_sz = lambda x: np.multiply(np.exp(-2*(np.sin( cdist(x[:,0].reshape(-1,1), Z[:,0].reshape(-1,1), 'minkowski', p=2)/(l[0]*l[0])))),np.exp(-2*(np.sin( cdist(x[:,1].reshape(-1,1), Z[:,1].reshape(-1,1),'sqeuclidean')/(l[1]*l[1])))))
                
            else:
                if kern=='RBF':
                    K_sz = lambda x: np.exp(-cdist(x, Z,'sqeuclidean')/(2*l*l))
                elif kern=='periodic':
                    K_sz = lambda x: np.exp(-2* ( np.sin( cdist(x, Z,'minkowski', p=1) / 2 )**2 ) /(l*l) )
                
    
            res1 = lambda x: K_sz(x) @ ( -np.linalg.inv( C*np.eye(Z.shape[0], Z.shape[0])+ 1e-3 * np.eye(Z.shape[0]) + Ksinv @ A   ) ) @ Ksinv@ (sumgradx_K[:,di] +sumgradG_K  )
            
                
            
    
    return res1   ### shape out N x dim



