import numpy as np
from scipy.special import beta, digamma, betainc

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

def get_window_boundaries(num_f):
    f = np.zeros(num_f)
    f[0] = 0.0
    f[1:] = get_psmc_times(num_f-1,0.5)
    return f

def get_freqs(num_f):
    v = get_window_boundaries(num_f)
    f = np.concatenate(((0,), (v[:-1]+v[1:])/2.0))
    return f

def get_lpf(params, x, window_boundaries = None):
    if window_boundaries is None:
        raise ValueError('must provide window_boundaries')
    v = window_boundaries
    lA, lB, expitz = params
    A = np.exp(lA)
    B = np.exp(lB)
    z = logistic(expitz)
    If = betainc(A,B, 2*v)
    lpf = np.concatenate(((np.log(z),), np.log(np.diff(If)*(1-z))))
    return lpf

def get_gradient(params, x, window_boundaries = None, eps = 1e-7):
    v = window_boundaries
    if v is None:
        raise ValueError('must provide window_boundaries')
    lA, lB, expitz = params
    A = np.exp(lA)
    B = np.exp(lB)
    z = logistic(expitz)
    If = betainc(A,B, 2*v)
    pf = np.diff(If)*(1-z)
    lpf = np.log(pf)

    # A
    Ape = np.exp(lA+eps)
    If_Ape = betainc(Ape,B,2*v)
    pf_Ape = np.diff(If_Ape)*(1-z)
    lpf_Ape = np.log(pf_Ape)
    Dlpf_Ape = (lpf_Ape-lpf)/eps
    Ame = np.exp(lA-eps)
    If_Ame = betainc(Ame,B,2*v)
    pf_Ame = np.diff(If_Ame)*(1-z)
    lpf_Ame = np.log(pf_Ame)
    Dlpf_Ame = (lpf-lpf_Ame)/eps
    Dlpf_A = (Dlpf_Ape+Dlpf_Ame)/2.0
    # B
    Bpe = np.exp(lB+eps)
    If_Bpe = betainc(A,Bpe,2*v)
    pf_Bpe = np.diff(If_Bpe)*(1-z)
    lpf_Bpe = np.log(pf_Bpe)
    Dlpf_Bpe = (lpf_Bpe-lpf)/eps
    Bme = np.exp(lB-eps)
    If_Bme = betainc(A,Bme,2*v)
    pf_Bme = np.diff(If_Bme)*(1-z)
    lpf_Bme = np.log(pf_Bme)
    Dlpf_Bme = (lpf-lpf_Bme)/eps
    Dlpf_B = (Dlpf_Bpe+Dlpf_Bme)/2.0
    # z
    zpe = logistic(expitz+eps)
    lpf_zpe = np.log(pf*(1-zpe))
    Dlpf_zpe = (lpf_zpe-lpf)/eps
    zme = logistic(expitz-eps)
    lpf_zme = np.log(pf*(1-zme))
    Dlpf_zme = (lpf-lpf_zme)/eps
    Dlpf_z = (Dlpf_zpe+Dlpf_zme)/2.0
    
    ret = np.zeros((np.asarray(x).shape[0], np.asarray(params).shape[0]))
    ret[1:,0] = Dlpf_A
    ret[1:,1] = Dlpf_B
    ret[1:,2] = Dlpf_z
    ret[0,2] = 1/(1+np.exp(expitz))
    return ret
