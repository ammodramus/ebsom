## cython: profile=True
## cython: linetrace=True
## cython: binding=True
## distutils: define_macros=CYTHON_TRACE_NOGIL=1
# cython: boundscheck=False
# cython: wraparound=False

cimport cython
cimport numpy as np
from libc.math cimport log, exp, abs, isnan, isfinite
from libc.stdio cimport printf
from libc.math cimport INFINITY, NAN
from doublevec cimport DoubleVec
from doubleveccounts cimport DoubleVecCounts
import beta_with_spikes_integrated as bws

import numpy as np
from scipy.special import logsumexp
import likelihood as lik
import afd
import util as ut

cdef inline double double_max(double a, double b): return a if a >= b else b
cdef inline double double_min(double a, double b): return a if a <= b else b

cdef inline int get_sign(double Xi, int l, int j):
    if Xi < 0 and l == j:
        return -1
    if Xi >= 0 and l == j:
        return 1
    if Xi < 0 and l != j:
        return 1
    if Xi >= 0 and l != j:
        return -1

cdef double logsumexp_double(double *x, int n, double maxx = NAN):
    if n == 0 or maxx == -INFINITY:
        return -INFINITY
    cdef double v
    cdef int i
    if isnan(maxx):
        maxx = x[0]
        for i in range(1,n):
            if x[i] > maxx:
                maxx = x[i]
    #printf("maxx: %f\n", maxx)
    cdef double S = 0.0
    S = 0.0
    #if False:
    if maxx > -2 and maxx < 2:
        # no logsumexp trick needed
        for i in range(n):
            S += exp(x[i])
        v = log(S)
    else:
        for i in range(n):
            S += exp(x[i]-maxx)
        v = maxx + log(S)
    if isnan(v):
        return -INFINITY
    else:
        return v

cdef double logsumexp_double_counts(double *x, unsigned int *counts, int n, double maxx = NAN):
    if n == 0 or maxx == -INFINITY:
        return -INFINITY
    cdef double v
    cdef int i
    if isnan(maxx):
        maxx = x[0]
        for i in range(1,n):
            if x[i] > maxx:
                maxx = x[i]
    #printf("maxx: %f\n", maxx)
    cdef double S = 0.0
    S = 0.0
    #if False:
    if maxx > -2 and maxx < 2:
        # no logsumexp trick needed
        for i in range(n):
            S += exp(x[i]) * counts[i]
        v = log(S)
    else:
        for i in range(n):
            S += exp(x[i]-maxx) * counts[i]
        v = maxx + log(S)
    if isnan(v):
        return -INFINITY
    else:
        return v

cdef void collect_alpha_delta_log_summands(
        int X_idx,
        int designated_outcome,
        int [:,::1] lo,
        double [:,::1] lpA,
        double [:,::1] lpa,
        double [:,::1] cm,
        bint is_major,
        double [:] lf,
        double [:] l1mf,
        list l_log_alpha_log_summands,
        list l_log_delta_log_summands):
    cdef:
        int i, j, k, nlo, lp_idx, count, observed_outcome
        double logsummand, c1, c2, logabsXi, obs_tlp, des_tlp, Xi, tlpa, tlpA
        double logsummand_nof, tlf, tl1mf
        DoubleVecCounts log_alpha_log_summands,
        DoubleVecCounts log_delta_log_summands

    cdef int nfs = lf.shape[0]
    nlo = lo.shape[0]
    for i in range(nlo):
        lp_idx = lo[i,0]
        Xi = cm[lp_idx, X_idx]
        logabsXi = log(abs(Xi))

        for observed_outcome in range(4):
            count = lo[i,observed_outcome+1]
            if count <= 0:
                continue
            if Xi == 0.0:
                for j in range(nfs):
                    log_alpha_log_summands = l_log_alpha_log_summands[j]
                    log_alpha_log_summands.append(-INFINITY, count)
                continue
                
            tlpA = lpA[lp_idx,observed_outcome]
            tlpa = lpa[lp_idx,observed_outcome]
            obs_tlp = tlpA if is_major else tlpa
            des_tlp = lpA[lp_idx,designated_outcome] if is_major else lpa[lp_idx,designated_outcome]
            logsummand_nof = obs_tlp
            logsummand_nof += logabsXi
            if observed_outcome == designated_outcome:
                #assert des_tlp == obs_tlp
                logsummand_nof += log(1-exp(des_tlp))
            else:
                logsummand_nof += des_tlp
            sign = get_sign(Xi, designated_outcome, observed_outcome)
            for j in range(nfs):
                tlf = lf[j]
                tl1mf = l1mf[j]
                c1 = tlf + tlpa
                c2 = tl1mf + tlpA
                M = double_max(c1,c2)
                m = double_min(c1,c2)

                logsummand = tl1mf if is_major else tlf
                logsummand -= M + log(1 + exp(m-M))
                logsummand += logsummand_nof

                # now add to log_alpha and log_delta
                log_alpha_log_summands = l_log_alpha_log_summands[j]
                log_delta_log_summands = l_log_delta_log_summands[j]
                if sign == 1:
                    log_alpha_log_summands.append(logsummand, count)
                else:
                    #assert sign == -1
                    log_delta_log_summands.append(logsummand, count)


def loc_ll_wrapper(args):
    return loc_ll(*args)

def loc_ll(
        np.ndarray[ndim=1,dtype=np.float64_t] params,
        double [:,::1] cm,
        dict logprobs,
        tuple locobs,
        bytes major,
        bytes minor,
        dict blims,
        double [:] lpf,
        double [:] lf,
        double [:] l1mf):

    if minor == 'N':
        return loc_ll_Nminor(params, cm, logprobs, locobs, major,
                minor, blims, lpf, lf, l1mf)

    cdef bytes rmajor, rminor
    rmajor, rminor = ut.rev_comp(major), ut.rev_comp(minor)

    cdef:
        int i, j, k, fidx, nobs, rowlen, nregs, nbetasperreg, nbetas, nfs, nlo
        int lowA, lowa, highA, higha, lp_idx, count, pp_all_len, bidx, pos
        list los, major_keys, lpAs, lowAs, highAs,
        list minor_keys, lpas, lowas, highas
        double tlpA, tlpa, M, m, c2, c3, val, logabsbf, tlf, tl1mf
        np.ndarray[ndim=2,dtype=np.double_t] lpA
        np.ndarray[ndim=2,dtype=np.double_t] lpa
        #double [:,::1] lpA
        #double [:,::1] lpa
        double [:] logaf
        int [:,::1] lo

    logaf = np.zeros(lpf.shape[0])

    rowlen = cm.shape[1]
    nfs = lf.shape[0]
    nregs = len(blims.keys())
    nbetasperreg = 3*rowlen
    nbetas = nregs*nbetasperreg
    nfs = lf.shape[0]

    # logaf := \log P(f) + \sum_i \log (fP(Yi|Xi,a) + (1-f)P(Yi|Xi,A) )
    logaf[:] = lpf[:]

    los = [locobs[0][0], locobs[0][1], locobs[1][0], locobs[1][1]]
    major_keys = [(major,1), (major,2), (rmajor,1), (rmajor,2)]
    lpAs = [logprobs[key] for key in major_keys]

    minor_keys = [(minor,1), (minor,2), (rminor,1), (rminor,2)]
    lpas = [logprobs[key] for key in minor_keys]

    # calculate the logaf's
    for i in range(len(los)):
        lo = los[i]
        lpA = lpAs[i]
        lpa = lpas[i]
        nlo = lo.shape[0]
        for j in range(nlo):
            lp_idx = lo[j,0]
            #assert lo.shape[1]-1 == 4
            for k in range(lo.shape[1]-1):
                count = lo[j,k+1]
                if count <= 0:   # should never be < 0
                    continue
                #assert k < 4
                tlpA = lpA[lp_idx,k]
                tlpa = lpa[lp_idx,k]
                # calculate the summands to add to logaf
                for fidx in range(nfs):
                    c2 = lf[fidx] + tlpa
                    c3 = l1mf[fidx] + tlpA
                    M = double_max(c2,c3)
                    m = double_min(c2,c3)
                    c4 = M + log(1 + exp(m-M))
                    logaf[fidx] += count*c4

    return logsumexp_double(&logaf[0], nfs)




def loc_ll_Nminor(params, cm, logprobs, locobs, major, minor, blims, lpf,
        lf, l1mf):
    if minor != 'N':
        raise ValueError('calling Nminor gradient when minor is not N')
    lls = []
    llps = []
    for newminor in 'ACGT':
        if newminor == major:
            continue
        ll = loc_ll(params, cm, logprobs, locobs, major, newminor,
                blims, lpf, lf, l1mf)
        lls.append(ll)
    lsell = logsumexp(lls)
    return lsell - np.log(3.0)

def get_args_debug(params, cm, lo, mm, blims, rowlen, freqs, windows, lf, l1mf, regs,
        num_f, num_pf_params):
    betas = params[:-num_pf_params]
    pf_params = params[-num_pf_params:]
    f = freqs
    v = windows
    logpf = bws.get_lpf(pf_params, f, windows)

    logprobs = {}
    X = cm
    for reg in regs:
        low, high = blims[reg]
        b = betas[low:high].reshape((rowlen,-1), order = 'F')
        Xb = np.column_stack((np.dot(X,b), np.zeros(X.shape[0])))
        Xb -= logsumexp(Xb, axis = 1)[:,None]
        logprobs[reg] = Xb

    args = []
    loc_info = []
    for ref in lo.keys():
        for bam in lo[ref].keys():
            for position in range(len(lo[ref][bam])):
                locobs = lo[ref][bam][position]
                major, minor = mm[ref][bam][position]
                major, minor = str(major), str(minor)
                if major == 'N':
                    continue
                args.append((params, cm, logprobs, locobs, major, minor, blims,
                    logpf, lf, l1mf))
                loc_info.append((ref, bam, position))
    return args, loc_info


@cython.wraparound(True)
def ll(params, cm, lo, mm, blims, rowlen, freqs, windows, lf, l1mf,
        regs, num_f, num_pf_params, pool):
    betas = params[:-num_pf_params]
    pf_params = params[-num_pf_params:]
    f = freqs
    v = windows
    logpf = bws.get_lpf(pf_params, f, windows)

    logprobs = {}
    X = cm
    for reg in regs:
        low, high = blims[reg]
        b = betas[low:high].reshape((rowlen,-1), order = 'F')
        Xb = np.column_stack((np.dot(X,b), np.zeros(X.shape[0])))
        Xb -= logsumexp(Xb, axis = 1)[:,None]
        logprobs[reg] = Xb

    args = []
    for ref in lo.keys():
        for bam in lo[ref].keys():
            for position in range(len(lo[ref][bam])):
                locobs = lo[ref][bam][position]
                major, minor = mm[ref][bam][position]
                major, minor = str(major), str(minor)
                if major == 'N':
                    continue
                args.append((params, cm, logprobs, locobs, major, minor, blims,
                    logpf, lf, l1mf))

    lls = pool.map(loc_ll_wrapper, args)
    ll = np.sum(lls)
    return ll
