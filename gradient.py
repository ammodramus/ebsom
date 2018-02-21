import numpy as np
from numba import jit
from scipy.special import logsumexp

import likelihood as lik
import afd
import util as ut


#@jit
def calc_loggh_p(X, obsidx, betas, low, high, tlp):
    gh_p = np.zeros_like(betas)
    gh_p[:,obsidx] += 1
    gh_p -= tlp[np.newaxis,:]
    gh_p *= X[:,np.newaxis]
    return np.log(gh_p).flatten(order = 'F')  # X-idxs goes down the rows here

#@jit
def calc_read_and_obs_logc_p(X, obsidx, betasA, betasa, lowA, highA, lowa,
        higha, tlpArow, tlparow, tlf, tl1mf, nparams):
    logc_p = np.zeros(nparams)
    tlpa = tlparow[obsidx]
    tlpA = tlpArow[obsidx]
    logd = tlf + tlpa
    logd_p = logd + calc_loggh_p(X, obsidx, betasa, lowa, higha, tlparow)
    loge = tl1mf + tlpA
    loge_p = loge + calc_loggh_p(X, obsidx, betasA, lowA, highA, tlpArow)
    M = max(logd, loge)
    logdpe = M + np.log(np.exp(logd-M) + np.exp(loge-M))
    # vector op here, should work...
    #M = np.maximum(logd_p, loge_p)
    #logd_ppe_p = M + np.log(np.exp(logd_p-M) + np.exp(loge_p-M))
    logc_p[lowA:highA] = loge_p - logdpe
    logc_p[lowa:higha] = logd_p - logdpe
    return logc_p

#@jit
def calc_logc_ps(betasA, betasa, lo, cm, lpA, lpa, tlf, tl1mf, lowA, highA,
        lowa, higha, nparams):
    nlo = lo[:,1:].sum()
    nlorows = lo.shape[0]
    rowlen = betasA.shape[0]
    logc_ps = np.zeros((nlo, nparams))
    cur_idx = 0
    for i in range(nlorows):
        tlo = lo[i]
        X = cm[tlo[0]]
        obscounts = tlo[1:-1]  # the last column doesn't have any params!
        lp_row_idx = tlo[0]
        tlpArow = lpA[lp_row_idx,:-1]  # be careful with this :-1... no params
        tlparow = lpa[lp_row_idx,:-1]
        for obsidx, count in enumerate(obscounts):
            if count <= 0:
                continue
            tlogc_ps = calc_read_and_obs_logc_p(X, obsidx, betasA,
                betasa, lowA, highA, lowa, higha, tlpArow, tlparow, tlf, tl1mf, nparams)
            next_idx = cur_idx + count
            logc_ps[cur_idx:next_idx] = tlogc_ps[np.newaxis,:]
            cur_idx = next_idx

    return logc_ps

#@jit
def calc_logb_p(params, blims, major, minor, rmajor, rminor, lp_maj_f1,
        lp_maj_f2, lp_min_f1, lp_min_f2, lp_maj_r1, lp_maj_r2, lp_min_r1,
        lp_min_r2, lo_f1, lo_f2, lo_r1, lo_r2, cm, tlf, tl1mf):
    # a constant, to be added to each element in the vector (except alpha values, watch out)

    # TODO!
    logb = calc_logb(params, blims, major, minor, rmajor, rminor, lp_maj_f1,
        lp_maj_f2, lp_min_f1, lp_min_f2, lp_maj_r1, lp_maj_r2, lp_min_r1,
        lp_min_r2, lo_f1, lo_f2, lo_r1, lo_r2, cm, tlf, tl1mf)

    rowlen = cm.shape[1]

    nparams = params.shape[0]
    logb_p = np.zeros_like(params)
    # calculate logc_p's for each read and each parameter, at first separating them into different regressions, then concatentating.
    # could also have a count for each read in each regression... but maybe easier to duplicate at first

    # forward, 1
    lo = lo_f1
    lpA = lp_maj_f1
    lpa = lp_min_f1
    lowA, highA = blims[(major, 1)]
    betasA = params[lowA:highA].reshape((rowlen, 3))
    lowa, higha = blims[(minor, 1)]
    betasa = params[lowa:higha].reshape((rowlen, 3))
    logc_ps_f1 = calc_logc_ps(betasA, betasa, lo, cm, lpA, lpa, tlf, tl1mf, lowA, highA, lowa, higha, nparams)


    # reverse, 1
    lo = lo_r1
    lpA = lp_maj_r1
    lpa = lp_min_r1
    lowA, highA = blims[(rmajor, 1)]
    betasA = params[lowA:highA].reshape((rowlen, 3))
    lowa, higha = blims[(rminor, 1)]
    betasa = params[lowa:higha].reshape((rowlen, 3))
    logc_ps_r1 = calc_logc_ps(betasA, betasa, lo, cm, lpA, lpa, tlf, tl1mf, lowA, highA, lowa, higha, nparams)

    # forward, 2
    lo = lo_f2
    lpA = lp_maj_f2
    lpa = lp_min_f2
    lowA, highA = blims[(major, 2)]
    betasA = params[lowA:highA].reshape((rowlen, 3))
    lowa, higha = blims[(minor, 2)]
    betasa = params[lowa:higha].reshape((rowlen, 3))
    logc_ps_f2 = calc_logc_ps(betasA, betasa, lo, cm, lpA, lpa, tlf, tl1mf, lowA, highA, lowa, higha, nparams)

    # reverse, 2
    lo = lo_r2
    lpA = lp_maj_r2
    lpa = lp_min_r2
    lowA, highA = blims[(rmajor, 1)]
    betasA = params[lowA:highA].reshape((rowlen, 3))
    lowa, higha = blims[(rminor, 1)]
    betasa = params[lowa:higha].reshape((rowlen, 3))
    logc_ps_r2 = calc_logc_ps(betasA, betasa, lo, cm, lpA, lpa, tlf, tl1mf, lowA, highA, lowa, higha, nparams)

    logc_ps = np.row_stack((logc_ps_f1, logc_ps_r1, logc_ps_f2, logc_ps_r2))
    assert logc_ps.shape[1] == params.shape[0]

    # logsumexp trick
    M = logc_ps.max(0)
    logb_p = M + np.log(np.sum(np.exp(logc_ps-M[np.newaxis,:])))
    return logb + logb_p


#@jit
def calc_loga_p(
        grad,
        params,
        blims,
        major,
        minor,
        rmajor,
        rminor,
        lp_maj_f1,
        lp_maj_f2,
        lp_min_f1,
        lp_min_f2,
        lp_maj_r1,
        lp_maj_r2,
        lp_min_r1,
        lp_min_r2,
        lo_f1,
        lo_f2,
        lo_r1,
        lo_r2,
        cm, logpf, lf, l1mf):
    '''
    needed for loga_p:
    params
    cm  (for X values)
    logobs
    logprobs
    '''
    # this will hold the values to be exp'ed in the logsumexp, eventually
    # want the logpf to change across the rows, the parameter index to change across the columns
    s = np.repeat(logpf.copy(), params.shape[0]).reshape((logpf.shape[0], -1))
    nj = logpf.shape[0]  # number of frequencies
    for i in range(nj):
        # for each f, calculate \log b'(\theta,f) for that f and the whole
        # vector of parameters (betas)
        tlf = lf[i]
        tl1mf = l1mf[i]
        # calc_logb_p returns the vector of \log b'(\theta,f)'s for all the betas
        # (could also call it like...)
        # calc_logb_p_2(s[i], params, blims, ...), so a temporary array need not be created
        s[i] += calc_logb_p(
                params,
                blims,
                major,
                minor,
                rmajor,
                rminor,
                lp_maj_f1,
                lp_maj_f2,
                lp_min_f1,
                lp_min_f2,
                lp_maj_r1,
                lp_maj_r2,
                lp_min_r1,
                lp_min_r2,
                lo_f1,
                lo_f2,
                lo_r1,
                lo_r2,
                cm, tlf, tl1mf)

    # (logsumexp on s here)
    M = s.max(0)

    loga_p = M + np.log(np.exp(s-M[np.newaxis,:]))

    return loga_p


def calc_loga_p_with_mm(grad, params, logprobs, cm, locobs, major, minor, blims, rowlen,
        freqs, breaks, logpf, lf, l1mf, regs):
    rmajor, rminor = ut.rev_comp(major), ut.rev_comp(minor)
    lp_maj_f1 = logprobs[(major,1)]
    lp_maj_f2 = logprobs[(major,2)]
    lp_maj_r1 = logprobs[(rmajor,1)]
    lp_maj_r2 = logprobs[(rmajor,2)]
    lo_f1 = locobs[0][0]
    lo_f2 = locobs[0][1]
    lo_r1 = locobs[1][0]
    lo_r2 = locobs[1][1]

    if minor != 'N':
        lp_min_f1 = logprobs[(minor,1)]
        lp_min_f2 = logprobs[(minor,2)]
        lp_min_r1 = logprobs[(rminor,1)]
        lp_min_r2 = logprobs[(rminor,2)]
        loga_p = calc_loga_p(
                grad,
                params,
                blims,
                major,
                minor,
                rmajor,
                rminor,
                lp_maj_f1,  # log-probabilities, from M-M product with cm and betas
                lp_maj_f2,
                lp_min_f1,
                lp_min_f2,
                lp_maj_r1,
                lp_maj_r2,
                lp_min_r1,
                lp_min_r2,
                lo_f1,  # locus observations
                lo_f2,
                lo_r1,
                lo_r2,
                cm,   # covariate matrix
                logpf,
                lf,
                l1mf)
    return loga_p

def gradient(params, ref, bam, position, cm, lo, mm, blims,
        rowlen, freqs, breaks, lf, l1mf, regs):

    betas = params[:-2]
    ab, ppoly = params[-2:]
    N = 1000
    logpf = np.log(afd.get_stationary_distribution_double_beta(
        freqs, breaks, N, ab, ppoly))

    ll = lik.single_locus_log_likelihood(params, ref, bam, position, cm, lo, mm, blims,
            rowlen, freqs, breaks, lf, l1mf, regs)
    logprobs = {}
    X = cm
    for reg in regs:
        low, high = blims[reg]
        b = betas[low:high].reshape((rowlen,-1))
        Xb = np.column_stack((np.dot(X,b), np.zeros(X.shape[0])))
        Xb -= logsumexp(Xb, axis = 1)[:,None]
        logprobs[reg] = Xb

    grad = np.zeros_like(params)

    locobs = lo[ref][bam][position]
    major, minor = mm[ref][bam][position]

    loga_p = calc_loga_p_with_mm(grad, params, logprobs, cm, locobs, major, minor, blims, rowlen,
            freqs, breaks, logpf, lf, l1mf, regs)
    import pdb; pdb.set_trace()

    return np.exp(loga_p - ll)
