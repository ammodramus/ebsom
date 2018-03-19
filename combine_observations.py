import numpy as np
from scipy.special import logsumexp
import util as ut
from collections import defaultdict

import deepdish as dd
import numpy as np
import numpy.random as npr
import afd
import gradient
import cygradient
import cylikelihood
import beta_with_spikes as bws
import cyglobal

def combine_locobs(locobs, mm):
    all_locobs = {}
    for base in 'ACGT':
        for read in [1,2]:
            key = (base, read)
            all_locobs[key] = defaultdict(lambda: np.zeros(4, dtype = np.int32))
    for ref in locobs.keys():
        for bam in locobs[ref].keys():
            for posidx, posobs in enumerate(locobs[ref][bam]):
                major, minor = mm[ref][bam][posidx]
                if major == 'N':
                    continue
                rmajor, rminor = ut.rev_comp(major), ut.rev_comp(minor)

                key = (major, 1)
                tpos = posobs[0][0]
                alo = all_locobs[key]
                for row in tpos:
                    cm_idx = row[0]
                    alo[cm_idx] += row[1:]
                key = (major, 2)
                tpos = posobs[0][1]
                alo = all_locobs[key]
                for row in tpos:
                    cm_idx = row[0]
                    alo[cm_idx] += row[1:]
                key = (rmajor, 1)
                tpos = posobs[1][0]
                alo = all_locobs[key]
                for row in tpos:
                    cm_idx = row[0]
                    alo[cm_idx] += row[1:]
                key = (rmajor, 2)
                tpos = posobs[1][1]
                alo = all_locobs[key]
                for row in tpos:
                    cm_idx = row[0]
                    alo[cm_idx] += row[1:]

    all_list = {}
    for base in 'ACGT':
        for read in [1,2]:
            regkey = (base, read)
            all_list[regkey] = []
            for cm_idx in all_locobs[regkey].keys():
                new_row = [cm_idx] + list(all_locobs[regkey][cm_idx])
                all_list[regkey].append(new_row)
            arr = np.array(all_list[regkey], dtype = np.int32)
            all_list[regkey] = arr
    return all_list

if __name__ == '__main__':
    cm, lo, mm = dd.io.load('empirical_onecm.h5')
    #cm, lo, mm = dd.io.load('testdata.h5')
    cm, cm_minmaxes = ut.normalize_covariates(cm)

    clo = combine_locobs(lo, mm)

    regkeys = [(b, r) for b in 'ACGT' for r in (1,2)]
    rowlen = cm.shape[1]
    blims = {}
    for i, reg in enumerate(regkeys):
        low = rowlen*3*i
        high = rowlen*3*(i+1)
        blims[reg] = (low, high)

    nbetas = len(regkeys)*3*rowlen
    npr.seed(0); betas = npr.uniform(-0.2,0.2, size=nbetas)
    num_pf_params = 3
    #a, b, z = -1, 0.5, -0.5
    #pars = np.concatenate((betas, (a,b,z)))
    #betas = np.loadtxt('global_params_1.txt')
    pars = betas

    #print cyglobal.calc_global_likelihood(pars, cm, clo, blims)
    import scipy.optimize as opt

    def ll_target(pars):
        v = -1*cyglobal.calc_global_likelihood(pars, cm, clo, blims)
        pstr = "\t".join([str(v)] + [str(el) for el in pars])
        #print pstr + '\n',
        return v
    grad_target = lambda pars: -1*cyglobal.calc_global_gradient(pars, cm, clo, blims)


    #from scipy.optimize import approx_fprime
    #approxgrad = approx_fprime(pars, ll_target, 1e-6)
    #numgrad = grad_target(pars)
    #for a, n in zip(approxgrad, numgrad):
    #    print '{}\t{}'.format(a,n)
    res = opt.minimize(ll_target, pars, method = 'L-BFGS-B', jac = grad_target)
    print res
