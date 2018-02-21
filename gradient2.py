import numpy as np
from numba import jit
from scipy.special import logsumexp
import likelihood as lik
import afd
import util as ut

def obs_partial_derivs(obs_idx, X, betas):
    '''
    returns a vector of partial derivatives, 3*rowlen, in F order, in the same
    order as betas, where betas[i,j] gives the i'th coefficient for outcome j
    '''

    logabsX = np.log(np.abs(x))
    Xb = np.zeros(4, dtype = np.complex64)
    np.dot(X.astype(np.complex64),betas.astype(np.complex64), out = Xb[:3])  # can use this 'out=' trick elsewhere instead of column_stack
    logsumexpXb = logsumexp(Xb)

    ret = np.full(betas.shape, -2.0*logsumexpXb)
    Xb_j = Xb[obs_idx]
    for l in range(3):
        if l == obs_idx:
            # 'logsubtractexp' trick
            a = logX + XB_j + logsumexpXb
            b = 2*XB_j + logX
            M = np.maximum(a,b)

            ret[:,l] += M + np.log(np.exp(a-M) - np.exp(b-M))
        else:
            ret[:,l] += Xb[l] + Xb_j + logX

    import pdb; pdb.set_trace()

    ret = ret.flatten(order = 'F')
    return ret

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
    lfcol = lf[:,np.newaxis]
    l1mfcol = l1mf[:,np.newaxis]

    # log P(Y). Note that a, above, corresponds to minor allele
    lpa_F1 = logprobs[(minor, 1)]
    lpA_F1 = logprobs[(major, 1)]
    lpa_F2 = logprobs[(minor, 2)]
    lpA_F2 = logprobs[(major, 2)]
    lpa_R1 = logprobs[(rminor, 1)]
    lpA_R1 = logprobs[(rmajor, 1)]
    lpa_R2 = logprobs[(rminor, 2)]
    lpA_R2 = logprobs[(rmajor, 2)]

    ###########################################################################
    # keep track of, for each f: 
    #     c1 = \sum_{i reads} \log ( f P(Yi|Xi,a,th) + (1-f)P(Yi|Xi,A,th) )
    #
    # c1 requires no logsumexp trick outside of considering each individual
    # read. The other sum, in log u, is exponentiated (that is, it's a
    # logsumexp instead), so have to store intermediate results in an array.
    # Also, each intermediate result will be an array (of partial derivatives),
    # so the intermediate results will actually be stored in a matrix S. There
    # will be an S, Smaj, for the major-allele parameters (betas) and an S for
    # the minor allele parameters, Smin
    #
    # Everything must be done once for each f, unless a three-dimensional
    # matrix is formed. Let's do the 3-D matrix.
    # 
    # The 3D matrices Smin and Smaj will be in C order, and priority should be
    # given to operations over the reads, so that will be the final column.
    # Then, operations over f, then operations over the parameter, beta, so the
    # dimensions will be (nbetas, nfs, nobs)
    #
    # But wait! There's more. P'() will be zero except within a regression
    # (possibly distinct for F1, F2, R1, R2). Also either P'(Yi|Xi,a,th) or
    # P'(Yi|Xi,A,th) will be zero. But still, can sum over all reads, with
    # zeros and all.
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
    S = np.zeros((nbetas, nfs, nobs))

    c1 = np.zeros(nfs)

    cur_obs = 0
    # forward, 1
    lo = locobs[0][0] # locobs for forward, 1
    nlo = lo.shape[0]
    lpA = logprobs[(major,1)]
    lpa = logprobs[(minor,1)]
    lowA, highA = blims[(major, 1)]
    lowa, higha = blims[(minor, 1)]
    betas_min = params[lowa:higha].reshape((-1,3))
    betas_maj = params[lowA:highA].reshape((-1,3))
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
            # c4 := log (f P(Y_i|a) + (1-f)P(Y_i|A))
            c4 = M + np.log(np.exp(c2) + np.exp(c3))
            c1 += count*c4  # the count is rolled into the sum here

            # add the vector of c4's too
            S[:,:,cur_obs:cur_obs+count] += c4[np.newaxis,:,np.newaxis]

            Xi = cm[lp_idx]
            assert Xi.shape[0] == rowlen

            pp_minor = obs_partial_derivs(j, Xi, betas_min) 
            # calculate log( f P'(Yi|Xi,a,lowbetas) ), add it to S[lowa:higha,:,cur_obs:cur_obs+count]
            pp_major = obs_partial_derivs(j, Xi, betas_maj)
            # calculate log( (1-f) P'(Yi|Xi,A,highbetas) ), add it to S[lowA:highA,:,cur_obs:cur_obs+count]

            cur_obs += count
            import pdb; pdb.set_trace()



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
        b = betas[low:high].reshape((rowlen,-1))
        Xb = np.column_stack((np.dot(X,b), np.zeros(X.shape[0])))
        Xb -= logsumexp(Xb, axis = 1)[:,None]
        logprobs[reg] = Xb

    grad = np.zeros_like(params)

    locobs = lo[ref][bam][position]
    major, minor = mm[ref][bam][position]

    return loc_gradient(params, cm, logprobs, locobs, major, minor, blims, logpf, lf, l1mf) 
