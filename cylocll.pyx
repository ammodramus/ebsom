#cython: boundscheck=False
#cython: wraparound=False
import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc
from libc.string cimport memcpy
from libc.math cimport log, exp
from libc.float cimport DBL_MAX

cdef double maxd(double *x, size_t n):
    cdef size_t i
    cdef double mx = -1.0*DBL_MAX
    for i in range(n):
        if x[i] > mx:
            mx = x[i]
    return mx

def calc_loc_ll(
        np.ndarray[ndim=2,dtype=np.double_t] lpA,
        np.ndarray[ndim=2,dtype=np.double_t] lpa,
        np.ndarray[ndim=2,dtype=np.int32_t] lo,
        np.ndarray[ndim=1,dtype=np.double_t] lpf, 
        np.ndarray[ndim=1,dtype=np.double_t] logf,
        np.ndarray[ndim=1,dtype=np.double_t] log1mf):
    cdef:
        double ll, c, d, M, e, Mp, s, el, tlpa, tlpA
        int nlo, nj, i, j, X_idx, k, count, nfreqs
        double[:,:] clpA, clpa
        double[:] clpf, clogf, clog1mf, ca, loclpsA, loclpsa
        double *a
        int[:,:] clo
        int[:] tlo
        double *p_lpf0
        double *p_a0

    clpA = lpA
    clpa = lpa
    clo = lo
    clpf = lpf
    clogf = logf
    clog1mf = log1mf

    p_lpf0 = &(clpf[0])

    nfreqs = lpf.shape[0]

    a = <double *>malloc(sizeof(double)*<size_t>nfreqs)
    p_a0 = a

    ll = 0.0
    nlo = lo.shape[0]
    nj = lpf.shape[0]
    for i in range(nlo):
        tlo = clo[i]
        X_idx = tlo[0]
        loclpsA = clpA[X_idx]
        loclpsa = clpa[X_idx]
        for k in range(4):
            count = tlo[k+1]
            if count <= 0:
                continue
            memcpy(p_lpf0, p_a0, sizeof(double)*nfreqs)
            tlpa = loclpsa[k]
            tlpA = loclpsA[k]
            Mp = -1.0*DBL_MAX
            for j in range(nj):
                c = clogf[j] + tlpa
                d = clog1mf[j] + tlpA
                M = max(c,d)
                e = M + log(exp(c-M) + exp(d-M))
                a[j] += e
                if a[j] > Mp:
                    Mp = a[j]
            s = 0
            for j in range(nfreqs):
                s += exp(a[j]-Mp)
            Mp += log(s)
            ll += count*Mp
    return ll
