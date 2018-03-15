## cython: profile=True
## cython: linetrace=True
## cython: binding=True
## cython: boundscheck=False
## cython: wraparound=False
## distutils: define_macros=CYTHON_TRACE_NOGIL=1

cimport numpy as np
import numpy as np
from scipy.special import logsumexp
from libc.math cimport log, exp

def calc_global_likelihood(
        np.ndarray[ndim=1,dtype=np.double_t] params,
        np.ndarray[ndim=2,dtype=np.double_t] X,
        dict obs,
        dict blims):

    cdef:
        int rowlen, obs_idx, j, obs_count, cm_idx
        dict reflo, bamlo
        int [:,:] regobs 
        double [:,:] logprobs
        double ll

    rowlen = X.shape[1]

    ll = 0.0
    for regkey in obs.keys():
        major, readnum = regkey
        if len(obs[regkey]) == 0:   # may not have observations for all regressions, esp in small cases
            continue
        regobs = obs[regkey]
        low, high = blims[regkey]
        b = params[low:high].reshape((rowlen,-1), order = 'F')
        Xb = np.column_stack((np.dot(X,b), np.zeros(X.shape[0])))
        Xb -= logsumexp(Xb, axis = 1)[:,None]
        logprobs = Xb
        for obs_idx in range(regobs.shape[0]):
            for j in range(4):
                obs_count = regobs[obs_idx,j+1]
                cm_idx = regobs[obs_idx,0]
                if obs_count > 0:
                    ll += logprobs[cm_idx, j]*obs_count
    return ll

def calc_global_gradient(
        np.ndarray[ndim=1,dtype=np.double_t] params,
        np.ndarray[ndim=2,dtype=np.double_t] X,
        dict obs,
        dict blims):

    cdef:
        int rowlen, obs_idx, j, obs_count, param_idx, low, high, param_outcome, param_row_idx, obs_outcome
        dict reflo, bamlo
        int [:,:] regobs 
        double [:,:] logprobs
        double x, prob_term
    rowlen = X.shape[1]
    grad_np = np.zeros(params.shape[0])
    cdef double [:] grad = grad_np
    for regkey in obs.keys():
        major, readnum = regkey
        regobs = obs[regkey]
        low, high = blims[regkey]
        b = params[low:high].reshape((rowlen,-1), order = 'F')
        Xb = np.column_stack((np.dot(X,b), np.zeros(X.shape[0])))
        Xb -= logsumexp(Xb, axis = 1)[:,None]
        logprobs = Xb
        for obs_idx in range(regobs.shape[0]):
            for obs_outcome in range(4):
                obs_count = regobs[obs_idx,obs_outcome]
                if obs_count > 0:
                    for param_idx in range(low,high):
                        param_outcome = (param_idx-low) // rowlen
                        param_row_idx = (param_idx-low) % rowlen
                        x = X[obs_idx,param_row_idx]
                        prob_term = -1 * exp(logprobs[obs_idx,param_outcome])
                        if obs_outcome == param_outcome:
                            prob_term += 1
                        grad[param_idx] += x*prob_term*obs_count
    return grad_np
