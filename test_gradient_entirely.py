import scipy.optimize as opt
import deepdish as dd
import numpy as np
import numpy.random as npr
import afd
import gradient
import cygradient
import cylikelihood
import beta_with_spikes_integrated as bws
import util
import sys

num_f = 100
f = bws.get_freqs(num_f)
v = bws.get_window_boundaries(num_f)
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
#betas = np.zeros(nbetas)
num_pf_params = 3
a, b, z = -1, 0.5, -0.5
pars = np.concatenate((betas, (a,b,z)))

import schwimmbad
pool = schwimmbad.MultiPool(10)
#if not pool.is_master():
#    pool.wait()
#    sys.exit(0)

def likefun(p):
    val = -1.0*cylikelihood.ll(p, cm, lo, all_majorminor, blims, rowlen, f, v, lf,
            l1mf, regkeys, num_f=100,num_pf_params=3,pool=pool)
    pstr = "\t".join([str(val)] + [str(el) for el in p])
    print pstr + '\n',
    return val

gradfun = lambda p: -1.0*cygradient.gradient(p, cm, lo, all_majorminor, blims, rowlen, f, v, lf, l1mf, regkeys, num_f=100,num_pf_params=3,pool=pool)

ll_pars = likefun(pars)
grad = gradfun(pars)
for g in grad:
    print g
print '---------'
ll_pars_eps = []
ngrads = []
eps = 1e-6
for i in range(pars.shape[0]-3, pars.shape[0]):
    pars_eps = pars.copy()
    pars[i] += eps
    eps_ll = likefun(pars_eps)
    ll_pars_eps.append(eps_ll)
    ngrads.append((eps_ll-ll_pars)/eps)
    print ngrads[-1]
print '------'
for ll_par_ep in ll_pars_eps:
    print ll_par_ep
