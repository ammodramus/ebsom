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

# badloci.txt is 1-based, these will be 0-based
badloci = np.loadtxt('badloci.txt').astype(np.int)-1
for bam in bam_fns:
    for bl in badloci:
        all_majorminor['chrM'][bam][bl] = ('N', 'N')
        lo['chrM'][bam][bl] = [[[],[]], [[],[]]]

regkeys = [(b, r) for b in 'ACGT' for r in (1,2)]
rowlen = cm.shape[1]
blims = {}
for i, reg in enumerate(regkeys):
    low = rowlen*3*i
    high = rowlen*3*(i+1)
    blims[reg] = (low, high)

nbetas = len(regkeys)*3*rowlen
#npr.seed(0); betas = npr.uniform(-0.1,0.1, size=nbetas)
betas = np.loadtxt('global_params_1.txt')
#betas = np.zeros(nbetas)
num_pf_params = 3
a, b, z = -1, 0.5, 4
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

res = opt.minimize(fun = likefun, x0 = pars, method = 'L-BFGS-B', jac = gradfun, options = {'maxiter': 5000})
#res = opt.minimize(fun = likefun, x0 = pars, method = 'Newton-CG', jac = gradfun, options = {'maxiter': 5000})
#res = opt.minimize(fun = likefun, x0 = pars, method = 'BFGS', jac = gradfun, options = {'maxiter': 5000})
resstr = '#' + '\t'.join([str(el) for el in res.x])
print resstr + '\n',
