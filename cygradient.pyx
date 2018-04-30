# cython: profile=True
# cython: linetrace=True
# cython: binding=True
# distutils: define_macros=CYTHON_TRACE_NOGIL=1
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
#cpdef void collect_alpha_delta_log_summands(
#def collect_alpha_delta_log_summands(
        int X_idx,
        int designated_outcome,
        int [:,::1] lo,
        #double [:,::1] lpA,
        #double [:,::1] lpa,
        #double [:,::1] cm,
        np.ndarray[ndim=2,dtype=np.float64_t] lpA,
        np.ndarray[ndim=2,dtype=np.float64_t] lpa,
        np.ndarray[ndim=2,dtype=np.float64_t] cm,
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
                continue  # rather than append -INFINITY to a vector to be logsumexp'd, just skip it
                #for j in range(nfs):
                #log_alpha_log_summands = l_log_alpha_log_summands[j]
                #log_alpha_log_summands.append(-INFINITY, count)
                
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
                if sign == 1:
                    log_alpha_log_summands = l_log_alpha_log_summands[j]
                    log_alpha_log_summands.append(logsummand, count)
                else:
                    log_delta_log_summands = l_log_delta_log_summands[j]
                    #assert sign == -1
                    log_delta_log_summands.append(logsummand, count)


def loc_gradient_wrapper(args):
    return loc_gradient(*args)

def loc_gradient(
        np.ndarray[ndim=1,dtype=np.float64_t] params,
        np.ndarray[dtype=np.float64_t,ndim=2] cm,
        dict logprobs,
        tuple locobs,
        bytes major,
        bytes minor,
        dict blims,
        double [:] lpf,
        double [:] lf,
        double [:] l1mf,
        np.ndarray[dtype=np.float64_t,ndim=2] Dlogpf,
        int num_pf_params,
        bint return_both = False):

    '''
    print '###############'
    print 'params', params
    print 'cm', cm
    print 'logprobs', logprobs
    print 'locobs', locobs
    print 'major', major
    print 'minor', minor
    print 'blims', blims
    print 'lpf', lpf
    print 'lf', lf
    print 'l1mf', l1mf
    print 'Dlogpf', Dlogpf
    print 'num_pf_params', num_pf_params
    print 'return_both', return_both
    print '####################################'
    '''
    if minor == 'N':
        return loc_gradient_Nminor(params, cm, logprobs, locobs, major,
                minor, blims, lpf, lf, l1mf, Dlogpf, num_pf_params)

    cdef bytes rmajor, rminor
    rmajor, rminor = ut.rev_comp(major), ut.rev_comp(minor)

    cdef np.ndarray[dtype=np.float64_t,ndim=1] grad_np = np.zeros(params.shape[0])
    cdef double [:] grad = grad_np

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

    grad[:] = 0.0
    cdef double logsumexplogaf = logsumexp_double(&logaf[0], nfs)

    # here, calculate the gradient for the distribution parameters
    logabs_Dlogpf = np.log(np.abs(Dlogpf))
    sign_Dlogpf = (Dlogpf >= 0)*2 - 1
    cdef DoubleVec log_pos_summands_Dlogpf = DoubleVec(lf.shape[0], 2)
    cdef DoubleVec log_neg_summands_Dlogpf = DoubleVec(lf.shape[0], 2)
    cdef double lse_pos, lse_neg
    for i in range(num_pf_params):
        log_pos_summands_Dlogpf.clear()
        log_neg_summands_Dlogpf.clear()
        for fidx in range(nfs):
            if sign_Dlogpf[fidx,i] == 1:
                log_pos_summands_Dlogpf.append(logaf[fidx] + logabs_Dlogpf[fidx,i])
                #print '({}) pos: {}'.format(fidx, logaf[fidx] + logabs_Dlogpf[fidx,i])
            else:
                assert sign_Dlogpf[fidx,i] == -1
                log_neg_summands_Dlogpf.append(logaf[fidx] + logabs_Dlogpf[fidx,i])
                #print '({}) neg: {}'.format(fidx, logaf[fidx] + logabs_Dlogpf[fidx,i])
        lse_pos = logsumexp_double(log_pos_summands_Dlogpf.data,
                                   log_pos_summands_Dlogpf.size,
                                   log_pos_summands_Dlogpf.cur_max)
        lse_neg = logsumexp_double(log_neg_summands_Dlogpf.data,
                                   log_neg_summands_Dlogpf.size,
                                   log_neg_summands_Dlogpf.cur_max)
        if lse_pos > lse_neg:
            grad[nbetas+i] = exp(lse_pos + np.log(1-np.exp(lse_neg-lse_pos)) - logsumexplogaf)
        else:
            grad[nbetas+i] = -exp(lse_neg + np.log(1-np.exp(lse_pos-lse_neg)) - logsumexplogaf)
        if isnan(grad[nbetas+i]):
            grad[nbetas+i] = 0.0

    # a_f is the same for all parameters. now have to calculate b_f for each
    # parameter.
    cdef list l_log_alpha_log_summands, l_log_delta_log_summands
    l_log_alpha_log_summands, l_log_delta_log_summands = [], []
    for fidx in range(nfs):
        l_log_alpha_log_summands.append(DoubleVecCounts(1024,2))
        l_log_delta_log_summands.append(DoubleVecCounts(1024,2))
    cdef DoubleVecCounts log_alpha_log_summands
    cdef DoubleVecCounts log_delta_log_summands
    cdef DoubleVec afbf_pos_log_summands = DoubleVec(256, 2)
    cdef DoubleVec afbf_neg_log_summands = DoubleVec(256, 2)
    cdef bint is_major, bf_is_pos, found
    cdef int X_idx, outcome
    cdef int idx, low, high, readno
    cdef double exp_pos, exp_neg, v, logdiff
    regressions = logprobs.keys()
    for reg in regressions:
        base, readno = reg
        low, high = blims[reg]
        for bidx in range(low,high):
            X_idx = (bidx-low) % rowlen
            outcome = (bidx-low) // rowlen
            #printf('%s %i %i %i %i %i %i\n', <bytes>base, readno, low, high,
            #        bidx, X_idx, outcome)
            #assert outcome < 3 and outcome >= 0
            #assert X_idx < rowlen
            afbf_pos_log_summands.clear()
            afbf_neg_log_summands.clear()
            found = False
            if (base == major or base == minor or base == rmajor or
                    base == rminor):
                found = True
            if not found:
                continue
            for fidx in range(nfs):
                log_alpha_log_summands = <DoubleVecCounts>l_log_alpha_log_summands[fidx]
                log_delta_log_summands = <DoubleVecCounts>l_log_delta_log_summands[fidx]
                log_alpha_log_summands.clear()
                log_delta_log_summands.clear()
            if base == major:
                # process as if base is major
                is_major = True
                lpA = logprobs[(major, readno)]
                lpa = logprobs[(minor, readno)]
                lo = locobs[0][readno-1]
                collect_alpha_delta_log_summands(X_idx, outcome,
                        lo, lpA, lpa, cm, is_major, lf, l1mf,
                        l_log_alpha_log_summands,
                        l_log_delta_log_summands)
            if base == minor:
                # process as if base is minor
                is_major = False
                lpA = logprobs[(major, readno)]
                lpa = logprobs[(minor, readno)]
                lo = locobs[0][readno-1]
                collect_alpha_delta_log_summands(X_idx, outcome,
                        lo, lpA, lpa, cm, is_major, lf, l1mf,
                        l_log_alpha_log_summands,
                        l_log_delta_log_summands)
            if base == rmajor:
                # process as if base is rmajor
                is_major = True
                lpA = logprobs[(rmajor, readno)]
                lpa = logprobs[(rminor, readno)]
                lo = locobs[1][readno-1]
                collect_alpha_delta_log_summands(X_idx, outcome,
                        lo, lpA, lpa, cm, is_major, lf, l1mf,
                        l_log_alpha_log_summands,
                        l_log_delta_log_summands)
            if base == rminor:
                # process as if base is rminor
                is_major = False
                lpA = logprobs[(rmajor, readno)]
                lpa = logprobs[(rminor, readno)]
                lo = locobs[1][readno-1]
                collect_alpha_delta_log_summands(X_idx, outcome,
                        lo, lpA, lpa, cm, is_major, lf, l1mf,
                        l_log_alpha_log_summands,
                        l_log_delta_log_summands)

                # calculate logabsbf[fidx]
            for fidx in range(nfs):
                log_alpha_log_summands = <DoubleVecCounts>l_log_alpha_log_summands[fidx]
                log_delta_log_summands = <DoubleVecCounts>l_log_delta_log_summands[fidx]
                if (log_delta_log_summands.size == 0 and
                        log_alpha_log_summands.size == 0):
                    continue
                log_alpha = logsumexp_double_counts(
                        log_alpha_log_summands.data,
                        log_alpha_log_summands.counts,
                        log_alpha_log_summands.size,
                        log_alpha_log_summands.cur_max)
                log_delta = logsumexp_double_counts(
                        log_delta_log_summands.data,
                        log_delta_log_summands.counts,
                        log_delta_log_summands.size,
                        log_delta_log_summands.cur_max)
                if log_alpha >= log_delta:
                    M = log_alpha
                    m = log_delta
                    bf_is_pos = True
                else:
                    M = log_delta
                    m = log_alpha
                    bf_is_pos = False
                logabsbf = M + log(1-exp(m-M))
                if isnan(logabsbf):
                    continue   # instead of appending -INFINITY to logsummands, just continue
                    #logabsbf = -INFINITY
                #printf("cy %i %i %f\n", bidx, fidx, logabsbf)
                #v = logaf[fidx] + logabsbf
                #printf("cy %i %i %f\n", bidx, fidx, v)
                #if not (isfinite(log_alpha) and isfinite(log_delta)):
                #    continue
                if bf_is_pos:
                    afbf_pos_log_summands.append(
                            logaf[fidx] + logabsbf)
                else:
                    afbf_neg_log_summands.append(
                            logaf[fidx] + logabsbf)
                # end fidx loop
            # now process the two exponentials that make up the gradient
            log_pos = logsumexp_double(afbf_pos_log_summands.data,
                    afbf_pos_log_summands.size,
                    afbf_pos_log_summands.cur_max) - logsumexplogaf
            log_neg = logsumexp_double(afbf_neg_log_summands.data,
                    afbf_neg_log_summands.size,
                    afbf_neg_log_summands.cur_max) - logsumexplogaf
            if log_pos > log_neg:
                logdiff = log_pos + log(1-exp(log_neg-log_pos))
                v = exp(logdiff)
                grad[bidx] = v if isfinite(v) else 0.0
            else:
                logdiff = log_neg + log(1-exp(log_pos-log_neg))
                v = -exp(logdiff)
                grad[bidx] = v if isfinite(v) else 0.0
            # end bidx loop
    if return_both:
        return logsumexplogaf, grad_np
    else:
        return grad_np




def loc_gradient_Nminor(params, cm, logprobs, locobs, major, minor, blims, lpf,
        lf, l1mf, logpf_grad, num_pf_params):
    if minor != 'N':
        raise ValueError('calling Nminor gradient when minor is not N')
    lls = []
    llps = []
    for newminor in 'ACGT':
        if newminor == major:
            continue
        ll, grad = loc_gradient(params, cm, logprobs, locobs, major, newminor,
            blims, lpf, lf, l1mf, logpf_grad, num_pf_params, return_both = True)
        lls.append(ll)
        llps.append(grad)

    nparams = llps[0].shape[0]
    lsell = logsumexp(lls)

    pos = np.zeros((nparams, 4))
    num_pos = np.zeros(nparams, dtype = np.int32)
    neg = np.zeros((nparams, 4))
    num_neg = np.zeros(nparams, dtype = np.int32)

    pos = np.zeros(4)
    neg = np.zeros(4)

    nlls = len(lls)

    ret = np.zeros(nparams)

    for i in range(nparams):
        pos[:] = 0.0
        neg[:] = 0.0
        num_pos = 0
        num_neg = 0
        for j in range(nlls):
            llp = llps[j][i]
            ll = lls[j]
            logabsllp = np.log(np.abs(llp))
            logsummand = ll + logabsllp - lsell
            if llp >= 0:
                pos[num_pos] = logsummand
                num_pos += 1
            else:
                neg[num_neg] = logsummand
                num_neg += 1

        a = logsumexp(pos[:num_pos]) if num_pos > 0 else -np.inf
        b = logsumexp(neg[:num_neg]) if num_neg > 0 else -np.inf
        ret[i] = np.exp(a) - np.exp(b)
    return ret


@cython.wraparound(True)
def loc_gradient_make_buffers(params, ref, bam, position, cm, lo, mm, blims,
        rowlen, freqs, lf, l1mf, regs, num_f, num_pf_params):

    betas = params[:-num_pf_params]
    pf_params = params[-num_pf_params:]
    f = freqs
    v = bws.get_window_boundaries(f.shape[0])
    logpf = bws.get_lpf(pf_params, f, v)

    logpf_grad = bws.get_gradient(pf_params,f,v)

    logprobs = {}
    X = cm
    for reg in regs:
        low, high = blims[reg]
        b = betas[low:high].reshape((rowlen,-1), order = 'F')
        Xb = np.column_stack((np.dot(X,b), np.zeros(X.shape[0])))
        Xb -= logsumexp(Xb, axis = 1)[:,None]
        logprobs[reg] = Xb

    locobs = lo[ref][bam][position]
    major, minor = mm[ref][bam][position]
    major, minor = bytes(major), bytes(minor)

    loc_grad = loc_gradient(params, cm, logprobs, locobs, major, minor, blims,
            logpf, lf, l1mf, logpf_grad, num_pf_params)
    return loc_grad


def get_args(lo, mm):
    args = []
    for ref in lo.keys():
        for bam in lo[ref].keys():
            for position in range(len(lo[ref][bam])):
                locobs = lo[ref][bam][position]
                major, minor = mm[ref][bam][position]
                if major == 'N':
                    continue
                found = False
                for i in [0,1]:
                    for j in [0,1]:
                        if locobs[i][j].shape[0] > 0:
                            found = True
                            break
                if not found:
                    continue
                args.append([locobs, major, minor])
    return args

# what we want:
#  - a function that takes a list of (locobs, major, minor) and returns the
#    gradient, calculated using the pool
def make_batch_gradient_func(cm, blims, lf, l1mf, num_pf_params, freqs, windows, regs, pool):
    rowlen = cm.shape[1]

    def f(params, argslist):
        betas = params[:-num_pf_params]
        pf_params = params[-num_pf_params:]
        f = freqs
        v = windows
        logpf = bws.get_lpf(pf_params, f, v)
        logpf_grad = bws.get_gradient(pf_params,f,v)
        logprobs = {}
        X = cm
        for reg in regs:
            low, high = blims[reg]
            b = betas[low:high].reshape((rowlen,-1), order = 'F')
            Xb = np.column_stack((np.dot(X,b), np.zeros(X.shape[0])))
            Xb -= logsumexp(Xb, axis = 1)[:,None]
            logprobs[reg] = Xb
        loc_gradient_args = [
                (params, cm, logprobs, locobs, major, minor, blims, logpf, lf,
                    l1mf, logpf_grad, num_pf_params) for locobs, major, minor
                in argslist
                ]
        grads = np.array(pool.map(loc_gradient_wrapper, loc_gradient_args))
        grad = np.sum(grads, axis = 0)
        return grads

    return f

def gradient(params, cm, lo, mm, blims, rowlen, freqs, windows, lf, l1mf,
        regs, num_f, num_pf_params, pool):
    betas = params[:-num_pf_params]
    pf_params = params[-num_pf_params:]
    f = freqs
    v = windows
    logpf = bws.get_lpf(pf_params, f, v)
    logpf_grad = bws.get_gradient(pf_params,f, v)

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
                if major == 'N':
                    continue
                args.append((params, cm, logprobs, locobs, major, minor, blims,
                    logpf, lf, l1mf, logpf_grad, num_pf_params))

    if len(args) == 0:
        grad = np.zeros_like(params)
    else:
        grads = pool.map(loc_gradient_wrapper, args)
        grad = np.sum(grads, axis = 0)
    return grad
