from __future__ import print_function
import datetime
import math
import numpy as np
from numba import jit
from scipy.special import logsumexp

import afd
import util as ut



#@jit
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

#@jit
def calc_loc_ll_wrong2(
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

@jit
def calc_loc_ll_cond_f_and_fr(lpA,  # the "major" log-probabilities for this orientation and read num.
                          lpa,      # " ", for minor
                          lo,     # the observations for this 
                          logf,   # now a double
                          log1mf):  # this also a double
    fll = 0.0
    Ma = -1e100  # very small number... DBL_MAX is ~1e308
    nlo = lo.shape[0]
    for j in range(nlo):
        X_idx = lo[j,0] # the index of this row in the covariate matrix and 
                       # log-probability matrices
        for k in range(4):
            count = lo[j,k+1]  # number of observations of this outcome with this covariate row
            if count <= 0:
                continue
            c = logf + lpa[X_idx,k]
            d = log1mf + lpA[X_idx,k]
            M = max(c,d)
            e = M + math.log(math.exp(c-M) + math.exp(d-M))
            fll += count*e
    return fll


@jit
def calc_loc_ll(lp_maj_f1,
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
                logpf, lf, l1mf):
    ll = 0.0
    nj = logpf.shape[0]  # number of frequencies
    Ma = -1e100  # very small number... DBL_MAX is ~1e308
    a = logpf.copy()
    for i in range(nj):
        tlf = lf[i]
        tl1mf = l1mf[i]
        tot_fll = 0.0
        # calculate the fll's for different fr/12's
        tot_fll += calc_loc_ll_cond_f_and_fr(
                lp_maj_f1,
                lp_min_f1,
                lo_f1,
                tlf,
                tl1mf)
        tot_fll += calc_loc_ll_cond_f_and_fr(
                lp_maj_f2,
                lp_min_f2,
                lo_f2,
                tlf,
                tl1mf)
        tot_fll += calc_loc_ll_cond_f_and_fr(
                lp_maj_r1,
                lp_min_r1,
                lo_r1,
                tlf,
                tl1mf)
        tot_fll += calc_loc_ll_cond_f_and_fr(
                lp_maj_r2,
                lp_min_r2,
                lo_r2,
                tlf,
                tl1mf)
        a[i] += tot_fll
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
    bamll = 0.0
    for i, locobs in enumerate(bamobs):
        major, minor = bam_mm[i]
        rmajor, rminor = ut.rev_comp(major), ut.rev_comp(minor)
        if major == 'N':
            continue
        locll = calc_loc_ll_with_mm(major, minor, logprobs, locobs, logpf, logf, log1mf)
        bamll += locll
    return bamll


def calc_loc_ll_with_mm(major, minor, logprobs, locobs, logpf, lf, l1mf):
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
        ll = calc_loc_ll(
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
                logpf, lf, l1mf)
    else:   # minor == 'N'
        avg_ll = np.zeros(3)
        i = 0
        for newminor in 'ACGT':
            if newminor == major:
                continue
            # recursive call with non-N minor
            avg_ll[i] = calc_loc_ll_with_mm(major, newminor, logprobs, locobs, logpf, lf, l1mf)
            i += 1
        assert i == 3
        ll = logsumexp(avg_ll) - np.log(3.0)   # average of 3
    return ll

def calc_likelihood(
        params, cm, lo, mm, blims, rowlen, freqs, breaks, lf, l1mf, regs,
        pool = None, printres = False):
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
                    if major == 'N':
                        continue
                    locll = calc_loc_ll_with_mm(major, minor, logprobs, locobs, logpf, lf, l1mf)
                    ll += locll
    if printres:
        ttime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ': '
        print(ttime + str(ll) + '\t' + '\t'.join([str(el) for el in params]))
    return ll
