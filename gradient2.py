import numpy as np
from numba import jit
from scipy.special import logsumexp
import likelihood as lik
import afd
import util as ut


def obs_partial_derivs(obs_idx, X, lP):
    '''
    Returns an array, of shape (3*rowlen,), of
        \log | P'(Yi|Xi,a,betas_min) |
    for each outcome, and an array of length (3*rowlen), containing the sign of
    the derivative.

    assumes that lP was calculated with four columns of betas, each of length
    rowlen
    '''

    rowlen = X.shape[0]
    logabsX = np.log(np.abs(X))
    ret = np.zeros((rowlen, 3), order = 'F')
    sign = np.zeros((rowlen, 3), order = 'F')
    ret += logabsX[:,np.newaxis] + lP[obs_idx]
    X_sign = (X >= 0)*2 - 1
    for l in range(3):
        if l == obs_idx:
            # logsumexp here would be equivalent to this
            ret[:,l] += np.log(1-np.exp(lP[l]))
            sign[:,l] = X_sign
        else:
            ret[:,l] += lP[l]
            sign[:,l] = -1*X_sign

    ret = ret.flatten(order = 'F')
    sign = sign.flatten(order = 'F')
    return ret, sign


def loc_gradient(params, cm, logprobs, locobs, major, minor, blims, lpf, lf, l1mf):
    '''
    going to iterate through f's, iterating through reads for each f.

    For each f, then need to calculate for each read (iterating over
    F1,F2,R1,R2):

        \log(f P(Y_i | X_i, a, params) + (1-f) P(Y_i | X_i, A, params),
    since this quantity is used three times for each read.

    the easiest way to do this is to populate a matrix the same size as
    logprobs with the appropriate mixture or log-mixture for each f, then read
    from that table. This will be slow, though, since this will calculate the
    probability for many more observations than are found at this locus.
    Instead, keep in a dictionary, indexed by the index in locobs?

    might also be able to do the same for P'(y...), etc.
    '''
    assert lpf.shape[0] == lf.shape[0] and lpf.shape[0] == l1mf.shape[0]
    rmajor, rminor = ut.rev_comp(major), ut.rev_comp(minor)

    ###########################################################################
    # a := \log f + \log P(Y_i | X_i, a, params)
    # b := \log (1-f) + \log P(Y_i | X_i, A, params)
    #
    # Need to calculate a and b for each F1,F2,R1,R2. Calculations will be done
    # using logsumexp trick instead of direct calculation. For each observation
    # in locobs, will store a and b in matrixes (say, A and B), where A[i,j] is
    # the value of a for the i'th f and the j'th outcome of the four possible,
    # corresponding to the entries in logprobs. B is similarly defined.
    #
    # Each matrix will be stored in a dict, since we want to calculate this
    # matrix only for the observations at this locus, not all the entries in
    # logprobs. The key will be the index in locobs (i.e., the first column in
    # locobs)
    ###########################################################################

    ###########################################################################
    # keep track of, for each f: 
    #     c1 = \sum_{i reads} \log ( f P(Yi|Xi,a,th) + (1-f)P(Yi|Xi,A,th) )
    #
    # c1 requires no logsumexp trick outside of considering each individual
    # read.
    #
    # For a given f and a given minor-allele regression parameter (indexed i),
    # have to keep a list of values of
    #
    #     \log f + \log |DP(Yi|Xi,a)| - \log(fP(Yi|Xi,a) + (1-f)P(Yi|Xi,A))
    #
    # where DP(Yi,Xi,a) is negative, and a list of the values
    #
    #     \log f + \log DP(Yi|Xi,a) - \log(fP(Yi|Xi,a) + (1-f)P(Yi|Xi,A))
    #
    # where DP(Yi,Xi,a) is positive.
    #
    # This also needs to be kept for the major-allele regression, but with
    #
    #     \log (1-f) + \log |DP(Yi|Xi,A)| - \log(fP(Yi|Xi,a) + (1-f)P(Yi|Xi,A)) 
    # and
    #
    #     \log (1-f) + \log DP(Yi|Xi,A) - \log(fP(Yi|Xi,a) + (1-f)P(Yi|Xi,A)) 
    #
    # respectively (note a -> A).
    #
    # The number of positive and negative values in these lists will be the
    # same for each f, but it will differ from parameter to parameter. So there
    # will be two counts, count_neg, and count_pos, for each parameter. This
    # means that there needs to be two arrays, S_pos, and S_neg, with shape
    # (n_fs, nobs) for each parameter. Store these in a list, and have
    # count_neg and count_pos in an array.
    ###########################################################################

    nobs = (
            locobs[0][0][:,1:].sum() + 
            locobs[0][1][:,1:].sum() + 
            locobs[1][0][:,1:].sum() + 
            locobs[1][1][:,1:].sum()
            )
    rowlen = cm.shape[1]
    nregs = len(blims.keys())
    nbetasperreg = 3*rowlen
    nbetas = nregs*nbetasperreg
    nfs = lf.shape[0]


    # c1 := \log P(f) + \sum_i \log (fP(Yi|Xi,a) + (1-f)P(Yi|Xi,A) )
    c1 = lpf.copy()

    los = [locobs[0][0], locobs[0][1], locobs[1][0], locobs[1][1]]
    major_keys = [(major,1), (major,2), (rmajor,1), (rmajor,2)]
    lpAs = [logprobs[key] for key in major_keys]
    lowAs = [blims[key][0] for key in major_keys]
    highAs = [blims[key][1] for key in major_keys]
    betas_maj = [params[lowA:highA].reshape((rowlen,3), order = 'F') for lowA, highA in zip(
        lowAs, highAs)]

    minor_keys = [(minor,1), (minor,2), (rminor,1), (rminor,2)]
    lpas = [logprobs[key] for key in minor_keys]
    lowas = [blims[key][0] for key in minor_keys]
    highas = [blims[key][1] for key in minor_keys]
    betas_min = [params[lowa:higha].reshape((rowlen,3), order = 'F') for lowa, higha in zip(
        lowas, highas)]

    S_neg = np.zeros((nbetas, nfs, nobs))
    S_pos = np.zeros((nbetas, nfs, nobs))

    count_neg = np.zeros(nbetas, dtype = np.int32)
    count_pos = np.zeros(nbetas, dtype = np.int32)

    for i, lo in enumerate(los):
        lpA = lpAs[i]
        lpa = lpas[i]
        lowA = lowAs[i]
        lowa = lowas[i]
        highA = highAs[i]
        higha = highas[i]
        betasA = betas_maj[i]
        betasa = betas_min[i]
        nlo = lo.shape[0]
        for j in range(nlo):
            lp_idx = lo[j,0]
            Xj = cm[lp_idx]
            for k, count in enumerate(lo[j,1:]):
                if count <= 0:   # should never be < 0
                    continue
                assert k < 4
                tlpA = lpA[lp_idx,k]
                tlpa = lpa[lp_idx,k]
                # calculate the summands to add to c1
                c2 = lf + tlpa
                c3 = l1mf + tlpA
                M = np.maximum(c2,c3)
                # c4 := log (f P(Y_i|a) + (1-f)P(Y_i|A)) for this (unique) read obs
                c4 = M + np.log(np.exp(c2-M) + np.exp(c3-M))
                c1 += count*c4  # the count is rolled into the sum here

                assert Xj.shape[0] == rowlen

                ################################################
                # pp_min will be 
                # 
                #    \log | P'(Yi|Xi,a,betas_min) | for each of the parameters.
                #
                # and sign_min will be the sign of this derivative for each
                # parameter (i.e., it will be a vector of length nbetasperreg)

                pp_min, sign_min = obs_partial_derivs(k, Xj, lpa[lp_idx]) 
                assert sign_min.shape[0] == nbetasperreg
                assert pp_min.shape[0] == 3*rowlen

                for l in range(pp_min.shape[0]):
                    bidx = lowa+l
                    pp, sign = pp_min[l], sign_min[l]
                    if sign > 0:
                        S_pos[bidx,:,count_pos[bidx]:count_pos[bidx]+count] = (
                                lf + pp - c4)[:,np.newaxis]  # note, adding c4 (def'd above) here.
                        count_pos[bidx] += count
                    else:
                        S_neg[bidx,:,count_neg[bidx]:count_neg[bidx]+count] = (
                                lf + pp - c4)[:,np.newaxis]  # note, adding c4 (def'd above) here.
                        count_neg[bidx] += count

                pp_maj, sign_maj = obs_partial_derivs(k, Xj, lpA[lp_idx])
                assert pp_maj.shape == sign_maj.shape
                #import pdb; pdb.set_trace()
                for l in range(pp_maj.shape[0]):
                    bidx = lowA+l
                    pp, sign = pp_maj[l], sign_maj[l]
                    if sign > 0:
                        S_pos[bidx, :,count_pos[bidx]:count_pos[bidx]+count] = (
                                l1mf + pp - c4)[:,np.newaxis]  # note, adding c4 (def'd above) here.
                        count_pos[bidx] += count
                    else:
                        S_neg[bidx, :,count_neg[bidx]:count_neg[bidx]+count] = (
                                l1mf + pp - c4)[:,np.newaxis]  # note, adding c4 (def'd above) here.
                        count_neg[bidx] += count


    ##### now, process ####

    # for each f and each parameter, we need to logsumexp the positive and the
    # negative. alpha is the logsumexp of the positive, delta is the logsumexp
    # of the negative

    alpha = np.zeros((nbetas, nfs), order = 'C')
    delta = np.zeros((nbetas, nfs), order = 'C')
    for i in range(nbetas):
        for j in range(nfs):
            if count_pos[i] > 0:
                alpha[i,j] = logsumexp(S_pos[i][j,:count_pos[i]])
            else:
                alpha[i,j] = -np.inf
            if count_neg[i] > 0:
                delta[i,j] = logsumexp(S_neg[i][j,:count_neg[i]])
            else:
                delta[i,j] = -np.inf

    log_max_ad = np.maximum(alpha, delta)
    log_min_ad = np.minimum(alpha, delta)

    sign_ad = (alpha > delta)*2-1

    M = log_max_ad
    # TODO define this, and check the logic
    logabsbf = M + np.log(1-np.exp(log_min_ad-log_max_ad))
    logabsbf[((~np.isfinite(log_max_ad)) & (~np.isfinite(log_min_ad)))] = (
            -np.inf)

    # for each parameter, need an array of the values of log a_f + log b_f,
    # where b_f is positive, and the values of log a_f + log |b_f|, where b_f
    # is negative. logsumexp each of these, exponentiate, and subtract

    # a_f is defined as log_f + \sum_i log( fP(Yi|Xi,a) + (1-f)P(Yi|Xi,A) ), or
    # c1

    logsumexpaf = logsumexp(c1)
    
    bf_pos = np.zeros(nbetas)
    bf_neg = np.zeros(nbetas)

    assert logabsbf.shape[0] == nbetas
    assert sign_ad.shape[0] == nbetas

    for i in range(nbetas):
        pos = sign_ad[i] > 0
        if np.any(pos):
            bf_pos[i] = logsumexp(logabsbf[i,pos] + c1[pos]) - logsumexpaf
        else:
            # TODO check the logic here, with the -np.inf
            bf_pos[i] = -np.inf
        if np.any(~pos):
            bf_neg[i] = logsumexp(logabsbf[i,~pos] + c1[~pos]) - logsumexpaf
        else:
            bf_neg[i] = -np.inf

    
    ret = np.exp(bf_pos) - np.exp(bf_neg)
    return ret


def loc_ll(params, cm, logprobs, locobs, major, minor, blims, lpf, lf, l1mf):
    '''
    going to iterate through f's, iterating through reads for each f.

    For each f, then need to calculate for each read (iterating over
    F1,F2,R1,R2):

        \log(f P(Y_i | X_i, a, params) + (1-f) P(Y_i | X_i, A, params),
    since this quantity is used three times for each read.

    the easiest way to do this is to populate a matrix the same size as
    logprobs with the appropriate mixture or log-mixture for each f, then read
    from that table. This will be slow, though, since this will calculate the
    probability for many more observations than are found at this locus.
    Instead, keep in a dictionary, indexed by the index in locobs?

    might also be able to do the same for P'(y...), etc.
    '''
    rmajor, rminor = ut.rev_comp(major), ut.rev_comp(minor)

    ###########################################################################
    # a := \log f + \log P(Y_i | X_i, a, params)
    # b := \log (1-f) + \log P(Y_i | X_i, A, params)
    #
    # Need to calculate a and b for each F1,F2,R1,R2. Calculations will be done
    # using logsumexp trick instead of direct calculation. For each observation
    # in locobs, will store a and b in matrixes (say, A and B), where A[i,j] is
    # the value of a for the i'th f and the j'th outcome of the four possible,
    # corresponding to the entries in logprobs. B is similarly defined.
    #
    # Each matrix will be stored in a dict, since we want to calculate this
    # matrix only for the observations at this locus, not all the entries in
    # logprobs. The key will be the index in locobs (i.e., the first column in
    # locobs)
    ###########################################################################

    ###########################################################################
    # keep track of, for each f: 
    #     c1 = \sum_{i reads} \log ( f P(Yi|Xi,a,th) + (1-f)P(Yi|Xi,A,th) )
    #
    # c1 requires no logsumexp trick outside of considering each individual
    # read.
    #
    # For a given f and a given minor-allele regression parameter (indexed i),
    # have to keep a list of values of
    #
    #     \log f + \log |DP(Yi|Xi,a)| - \log(fP(Yi|Xi,a) + (1-f)P(Yi|Xi,A))
    #
    # where DP(Yi,Xi,a) is negative, and a list of the values
    #
    #     \log f + \log DP(Yi|Xi,a) - \log(fP(Yi|Xi,a) + (1-f)P(Yi|Xi,A))
    #
    # where DP(Yi,Xi,a) is positive.
    #
    # This also needs to be kept for the major-allele regression, but with
    #
    #     \log (1-f) + \log |DP(Yi|Xi,A)| - \log(fP(Yi|Xi,a) + (1-f)P(Yi|Xi,A)) 
    # and
    #
    #     \log (1-f) + \log DP(Yi|Xi,A) - \log(fP(Yi|Xi,a) + (1-f)P(Yi|Xi,A)) 
    #
    # respectively (note a -> A).
    #
    # The number of positive and negative values in these lists will be the
    # same for each f, but it will differ from parameter to parameter. So there
    # will be two counts, count_neg, and count_pos, for each parameter. This
    # means that there needs to be two arrays, S_pos, and S_neg, with shape
    # (n_fs, nobs) for each parameter. Store these in a list, and have
    # count_neg and count_pos in an array.
    ###########################################################################

    rowlen = cm.shape[1]
    nregs = len(blims.keys())
    nbetasperreg = 3*rowlen
    nbetas = nregs*nbetasperreg
    nfs = lf.shape[0]

    c1 = lpf.copy()

    # forward, 1
    lo = locobs[0][0] # locobs for forward, 1
    nlo = lo.shape[0]
    lpA = logprobs[(major,1)]
    lpa = logprobs[(minor,1)]
    lowA, highA = blims[(major, 1)]
    lowa, higha = blims[(minor, 1)]
    betas_min = params[lowa:higha].reshape((rowlen,3), order = 'F')
    betas_maj = params[lowA:highA].reshape((rowlen,3), order = 'F')
    for i in range(nlo):
        lp_idx = lo[i,0]
        for j, count in enumerate(lo[i,1:]):
            if count <= 0:   # should never be < 0
                continue
            tlpA = lpA[lp_idx,j]
            tlpa = lpa[lp_idx,j]
            # calculate the summands to add to c1
            c2 = lf + tlpa
            c3 = l1mf + tlpA
            M = np.maximum(c2,c3)
            # c4 := log (f P(Y_i|a) + (1-f)P(Y_i|A)) for this (unique) read obs
            c4 = M + np.log(np.exp(c2-M) + np.exp(c3-M))
            c1 += count*c4  # the count is rolled into the sum here

    lo = locobs[0][1] # locobs for forward, 2
    nlo = lo.shape[0]
    lpA = logprobs[(major,2)]
    lpa = logprobs[(minor,2)]
    lowA, highA = blims[(major, 2)]
    lowa, higha = blims[(minor, 2)]
    betas_min = params[lowa:higha].reshape((-1,3), order = 'F')
    betas_maj = params[lowA:highA].reshape((-1,3), order = 'F')
    for i in range(nlo):
        lp_idx = lo[i,0]
        for j, count in enumerate(lo[i,1:]):
            if count <= 0:   # should never be < 0
                continue
            tlpA = lpA[lp_idx,j]
            tlpa = lpa[lp_idx,j]
            # calculate the summands to add to c1
            c2 = lf + tlpa
            c3 = l1mf + tlpA
            M = np.maximum(c2,c3)
            # c4 := log (f P(Y_i|a) + (1-f)P(Y_i|A)) for this (unique) read obs
            c4 = M + np.log(np.exp(c2-M) + np.exp(c3-M))
            c1 += count*c4  # the count is rolled into the sum here


    lo = locobs[1][0] # locobs for reverse, 1
    nlo = lo.shape[0]
    lpA = logprobs[(rmajor,1)]
    lpa = logprobs[(rminor,1)]
    lowA, highA = blims[(rmajor, 1)]
    lowa, higha = blims[(rminor, 1)]
    betas_min = params[lowa:higha].reshape((-1,3), order = 'F')
    betas_maj = params[lowA:highA].reshape((-1,3), order = 'F')
    for i in range(nlo):
        lp_idx = lo[i,0]
        for j, count in enumerate(lo[i,1:]):
            if count <= 0:   # should never be < 0
                continue
            tlpA = lpA[lp_idx,j]
            tlpa = lpa[lp_idx,j]
            # calculate the summands to add to c1
            c2 = lf + tlpa
            c3 = l1mf + tlpA
            M = np.maximum(c2,c3)
            # c4 := log (f P(Y_i|a) + (1-f)P(Y_i|A)) for this (unique) read obs
            c4 = M + np.log(np.exp(c2-M) + np.exp(c3-M))
            c1 += count*c4  # the count is rolled into the sum here


    lo = locobs[1][1] # locobs for reverse, 2
    nlo = lo.shape[0]
    lpA = logprobs[(rmajor,2)]
    lpa = logprobs[(rminor,2)]
    lowA, highA = blims[(rmajor, 2)]
    lowa, higha = blims[(rminor, 2)]
    betas_min = params[lowa:higha].reshape((-1,3), order = 'F')
    betas_maj = params[lowA:highA].reshape((-1,3), order = 'F')
    for i in range(nlo):
        lp_idx = lo[i,0]
        for j, count in enumerate(lo[i,1:]):
            if count <= 0:   # should never be < 0
                continue
            tlpA = lpA[lp_idx,j]
            tlpa = lpa[lp_idx,j]
            # calculate the summands to add to c1
            c2 = lf + tlpa
            c3 = l1mf + tlpA
            M = np.maximum(c2,c3)
            # c4 := log (f P(Y_i|a) + (1-f)P(Y_i|A)) for this (unique) read obs
            c4 = M + np.log(np.exp(c2-M) + np.exp(c3-M))
            c1 += count*c4  # the count is rolled into the sum here


    ##### now, process ####

    # for each f and each parameter, we need to logsumexp the positive and the
    # negative. alpha is the logsumexp of the positive, delta is the logsumexp
    # of the negative

    logsumexpaf = logsumexp(c1)
    
    return logsumexpaf


def gradient(params, ref, bam, position, cm, lo, mm, blims,
        rowlen, freqs, breaks, lf, l1mf, regs):

    betas = params[:-2]
    ab, ppoly = params[-2:]
    N = 1000
    logpf = np.log(afd.get_stationary_distribution_double_beta(
        freqs, breaks, N, ab, ppoly))

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

    return loc_gradient(params, cm, logprobs, locobs, major, minor, blims, logpf, lf, l1mf) 

def grad_locus_log_likelihood(params, ref, bam, position, cm, lo, mm, blims,
        rowlen, freqs, breaks, lf, l1mf, regs):
    betas = params[:-2]
    ab, ppoly = params[-2:]
    N = 1000
    logpf = np.log(afd.get_stationary_distribution_double_beta(
        freqs, breaks, N, ab, ppoly))
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

    return loc_ll(params, cm, logprobs, locobs, major, minor, blims, logpf, lf, l1mf) 
