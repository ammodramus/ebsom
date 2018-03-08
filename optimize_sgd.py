import scipy.optimize as opt
import deepdish as dd
import numpy as np
import numpy.random as npr
import afd
import gradient
import cygradient
import cylikelihood
import beta_with_spikes as bws
import util
import sys

num_f = 100
f = bws.get_freqs(num_f)
lf = np.log(f)
l1mf = np.log(1-f)

cm, lo, all_majorminor = dd.io.load('empirical_onecm.h5')
cm, cm_minmaxes = util.normalize_covariates(cm)

for m, M in cm_minmaxes:
    print '# mM {}\t{}'.format(m,M)

bam_fns = lo['chrM'].keys()

lo = util.sort_lo(lo)

regkeys = [(b, r) for b in 'ACGT' for r in (1,2)]
rowlen = cm.shape[1]
blims = {}
for i, reg in enumerate(regkeys):
    low = rowlen*3*i
    high = rowlen*3*(i+1)
    blims[reg] = (low, high)

nbetas = len(regkeys)*3*rowlen
npr.seed(0); betas = npr.uniform(-0.1,0.1, size=nbetas)
num_pf_params = 3
a, b, z = -1, 0.5, -0.5
pars = np.concatenate((betas, (a,b,z)))

import schwimmbad
pool = schwimmbad.MultiPool(3)

def likefun(p):
    v = -1.0*cylikelihood.ll(p, cm, lo, all_majorminor, blims, rowlen, f, lf,
            l1mf, regkeys, num_f=100,num_pf_params=3,pool=pool)
    pstr = "\t".join([str(v)] + [str(el) for el in p])
    print pstr + '\n',
    return v

arglist = cygradient.get_args(lo, all_majorminor)
gradfun = cygradient.make_batch_gradient_func(cm, blims, lf, l1mf, num_pf_params, f, regkeys, pool)
grad_target = lambda pars, arglist: -1.0*gradfun(pars, arglist)

alpha = 0.01
niter = 10000
batch_size = 100

W = pars.copy()
Wprev = W.copy()
Wgrad_prev = np.zeros_like(Wprev)

num_args = len(arglist)
split_at = np.arange(0, num_args, batch_size)[1:]

while True:
    permuted_args = npr.permutation(arglist)
    batches = np.array_split(permuted_args, split_at)
    for batch in batches:
        Wgrad = grad_target(W, batch)

        # if Wgrad is nan, take the previous parameters and reupdate them with
        # a learning rate divided by two.

        while np.any(np.isnan(Wgrad)):
            c = 2.0
            while True:
                W[:] = Wprev[:] + Wgrad_prev * -alpha/c
                Wgrad = grad_target(W, batch)
                if not np.any(np.isnan(Wgrad)):
                    break
                c *= 2.0
        Wgrad_prev = Wgrad[:]
        Wprev[:] = W[:]
        W += -alpha * Wgrad
        print W
