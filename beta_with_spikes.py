import numpy as np
from scipy.special import beta, digamma

def logistic(x):
    return 1.0/(1.0+np.exp(-x))

def Dlogistic(x):
    return np.exp(x)/((1+np.exp(x))**2.0)


def get_psmc_times(n, tmax):
    t = 0.1*(np.exp(
        np.arange(1,n+1, dtype = np.float)/n * np.log(1 + 10*tmax))-1)
    return t

def get_freqs(num_f):
    f = np.zeros(num_f)
    f[0] = 0.0
    f[1:] = get_psmc_times(num_f-1,0.5)
    f = f[:-1]  # frequency = 0.5 causes numerical problems: log(0) == -inf
    return f

def get_lpf(params, x):
    A, B, z = params
    eA = np.exp(A)
    eB = np.exp(B)
    lz = logistic(z)
    lpf = np.zeros_like(x)
    lpf[x==0.0] = np.log(lz)
    good = (0<x)&(x<=0.5)
    xg = x[good]
    lpf[good] = eA*np.log(2) + (eB-1)*np.log(1-2*xg) + (eA-1)*np.log(xg) + np.log(1-lz) - np.log(beta(eA,eB))
    return lpf

def get_gradient(params, x):
    # returns a gradient for each value of x, so a matrix M with shape
    # (nx,nparams)
    A, B, z = params
    eA = np.exp(A)
    eB = np.exp(B)
    lz = logistic(z)
    good = (0<x)&(x<=0.5)
    xg = x[good]
    M = np.zeros((x.shape[0],3), order = 'F')
    # for A
    M[good,0] = eA*(np.log(2*xg)-digamma(eA)+digamma(eA+eB))
    # for B
    M[good,1] = eB*(np.log(1-2*xg)-digamma(eB)+digamma(eA+eB))
    # for z
    M[good,2] = Dlogistic(z)/(lz-1)
    M[x==0.0,2] = Dlogistic(z)
    return M
