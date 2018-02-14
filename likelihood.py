import math
import numpy as np
from numba import jit
from scipy.special import logsumexp

import afd
import util as ut


@jit
def calc_loc_ll_wrong(
        lpA, lpa, lo, lpf, logf, log1mf):
    ll = 0.0
    nlo = lo.shape[0]  # number of observations at this locus
    nj = lpf.shape[0]  # number of frequencies
    for i in range(nlo):
        tlo = lo[i]    # this observation's row
        X_idx = tlo[0] # the index of this row in the covariate matrix and 
                       # log-probability matrices
        loclpsA = lpA[X_idx]  # the row of log-probabilities for the major allele
        loclpsa = lpa[X_idx]  # " " minor " "
        for k in range(4):
            count = tlo[k+1]  # number of observations of this outcome with this covariate row
            if count <= 0:
                continue
            a = lpf.copy()  # part of the sum/integration below ... this part
                            # is the same for every locus
            tlpa = loclpsa[k]
            tlpA = loclpsA[k]
            for j in range(nj):  # here summing over major and minor true
                c = logf[j] + tlpa # alleles
                d = log1mf[j] + tlpA
                M = max(c,d)
                e = M + math.log(math.exp(c-M) + math.exp(d-M))
                a[j] += e
            Mp = a.max()
            s = 0
            for el in a:  # here, logsumexp (summing) over frequencies
                s += math.exp(el-Mp)
            Mp += math.log(s)  # Mp here becomes the ll for this observation
            ll += count*Mp  # multiply by the count of this observation
    return ll


@jit
def calc_loc_ll(
        lpA, lpa, lo, lpf, logf, log1mf):
    ll = 0.0
    nlo = lo.shape[0]  # number of observations at this locus
    nj = lpf.shape[0]  # number of frequencies
    Ma = -1e100  # very small number... DBL_MAX is ~1e308
    a = lpf.copy()
    for i in range(nj):
        tlf = logf[i]
        tl1mf = log1mf[i]
        # could change the order here... profile
        fll = 0.0
        for j in range(nlo):
            X_idx = lo[j,0] # the index of this row in the covariate matrix and 
                           # log-probability matrices
            for k in range(4):
                count = lo[j,k+1]  # number of observations of this outcome with this covariate row
                if count <= 0:
                    continue
                c = tlf + lpa[X_idx,k]
                d = tl1mf + lpA[X_idx,k]
                M = max(c,d)
                e = M + math.log(math.exp(c-M) + math.exp(d-M))
                fll += count*e
        a[i] += fll
        if a[i] > Ma:
            Ma = a[i]
    locll = 0   # logsumexp routine here
    for el in a:
        locll += math.exp(el-Ma)
    locll = math.log(locll) + Ma
    return locll

def calc_bam_ll(data):
    avg_ll = np.zeros(3)
    params, bamobs, bam_mm, logprobs, logpf, logf, log1mf = data
    ll = 0.0
    for i, locobs in enumerate(bamobs):
        major, minor = bam_mm[i]
        rmajor, rminor = ut.rev_comp(major), ut.rev_comp(minor)
        regkeys = (
            ((major, 1), (minor, 1), (0,0)),
            ((major, 2), (minor, 2), (0,1)),
            ((rmajor, 1), (rminor, 1), (1,0)),
            ((rmajor, 2), (rminor, 2), (1,1))
            )
        if major == 'N':
            continue
        for majreg, minreg, (fridx, ridx) in regkeys:
            lp_maj = logprobs[majreg]
            tlo = locobs[fridx][ridx]
            if minor != 'N':
                # both major and minor are non-missing
                lp_min = logprobs[minreg]
                ll += calc_loc_ll(lp_maj, lp_min, tlo, logpf, logf, log1mf)
            else:  # major is non-missing, minor is missing, average over the others
                tmaj, tmin = majreg[0], minreg[0]
                tridx = minreg[1]
                i = 0
                for base in 'ACGT':
                    if base == tmaj:
                        continue
                    minreg = (base, tridx)
                    lp_min = logprobs[minreg]
                    avg_ll[i] = calc_loc_ll(
                            lp_maj, lp_min, tlo, logpf, logf, log1mf)
                    i += 1
                assert i == 3
                ll += logsumexp(avg_ll) - np.log(3.0)
    return ll


def calc_likelihood(
        params, cm, lo, mm, blims, rowlen, freqs, breaks, lf, l1mf, regs,
        pool = None):
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
        Xb = np.dot(X,b)
        Xb -= logsumexp(Xb, axis = 1)[:,None]
        logprobs[reg] = np.column_stack((Xb, np.zeros(Xb.shape[0])))
    chroms = lo.keys()
    ll = 0.0
    avg_ll = np.zeros(3)
    for chrom, chromobs in lo.iteritems():
        chrom_mm = mm[chrom]
        if pool is not None:
            tasks = []
            for bam_fn, bamobs in chromobs.iteritems():
                bam_mm = chrom_mm[bam_fn]
                data = (params, bamobs, bam_mm, logprobs, logpf, lf, l1mf)
                tasks.append(data)
            lls = list(pool.map(calc_bam_ll, tasks))
            ll = np.sum(lls)

        else:
            for bam_fn, bamobs in chromobs.iteritems():
                print('working on', bam_fn)
                bam_mm = chrom_mm[bam_fn]
                for i, locobs in enumerate(bamobs):
                    major, minor = bam_mm[i]
                    rmajor, rminor = ut.rev_comp(major), ut.rev_comp(minor)
                    regkeys = (
                        ((major, 1), (minor, 1), (0,0)),
                        ((major, 2), (minor, 2), (0,1)),
                        ((rmajor, 1), (rminor, 1), (1,0)),
                        ((rmajor, 2), (rminor, 2), (1,1))
                        )
                    if major == 'N':
                        continue
                    for majreg, minreg, (fridx, ridx) in regkeys:
                        lp_maj = logprobs[majreg]
                        tlo = locobs[fridx][ridx]
                        if minor != 'N':
                            # both major and minor are non-missing
                            lp_min = logprobs[minreg]
                            ll += calc_loc_ll(lp_maj, lp_min, tlo, logpf, lf, l1mf)
                        else:  # major is non-missing, minor is missing, average over the others
                            tmaj, tmin = majreg[0], minreg[0]
                            tridx = minreg[1]
                            i = 0
                            for base in 'ACGT':
                                if base == tmaj:
                                    continue
                                minreg = (base, tridx)
                                lp_min = logprobs[minreg]
                                avg_ll[i] = calc_loc_ll(
                                        lp_maj, lp_min, tlo, logpf, lf, l1mf)
                                i += 1
                            assert i == 3
                            ll += logsumexp(avg_ll) - np.log(3.0)
    return ll
