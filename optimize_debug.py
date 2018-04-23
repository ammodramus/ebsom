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
import argparse

parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--start-locus', type = int, help = 'index of first locus to consider for debugging (zero-based)')
parser.add_argument('--end-locus', type = int, help = 'index of last locus to consider for debugging (zero-based, inclusive)')
parser.add_argument('--num-processes', type = int, default = 1)
parser.add_argument('--init-params', type = str, help = 'filename of file containing list of initial parameters')
args = parser.parse_args()


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

if args.start_locus is not None:
    if args.end_locus is None:
        raise Exception('must provide both --start-locus and --end-locus')
    bam = bam_fns[0]  # assume we want the first bam?
    lotmp = {}
    lotmp['chrM'] = {}
    lotmp['chrM'][bam] = []
    mmtmp = {}
    mmtmp['chrM'] = {}
    mmtmp['chrM'][bam] = []
    for locidx in range(args.start_locus, args.end_locus+1):
        lotmp['chrM'][bam].append(lo['chrM'][bam][locidx])
        mmtmp['chrM'][bam].append(all_majorminor['chrM'][bam][locidx])
    lo = lotmp
    all_majorminor = mmtmp

regkeys = [(b, r) for b in 'ACGT' for r in (1,2)]
rowlen = cm.shape[1]
blims = {}
for i, reg in enumerate(regkeys):
    low = rowlen*3*i
    high = rowlen*3*(i+1)
    blims[reg] = (low, high)

nbetas = len(regkeys)*3*rowlen
num_pf_params = 3
if args.init_params is not None:
    pars = np.loadtxt(args.init_params)
else:
    nbetas = len(regkeys)*3*rowlen
    #betas = npr.uniform(-0.1,0.1, size=nbetas)
    betas = np.zeros(nbetas)
    num_pf_params = 3
    a, b, z = -1, 0.5, 4
    pars = np.concatenate((betas, (a,b,z)))

import schwimmbad
if args.num_processes > 1:
    pool = schwimmbad.MultiPool(args.num_processes)
else:
    pool = schwimmbad.SerialPool()

def likefun(p):
    val = -1.0*cylikelihood.ll(p, cm, lo, all_majorminor, blims, rowlen, f, v, lf,
            l1mf, regkeys, num_f=100,num_pf_params=3,pool=pool)
    pstr = "\t".join([str(val)] + [str(el) for el in p])
    print pstr + '\n',
    return val

gradfun = lambda p: -1.0*cygradient.gradient(p, cm, lo, all_majorminor, blims, rowlen, f, v, lf, l1mf, regkeys, num_f=100,num_pf_params=3,pool=pool)

res = opt.minimize(fun = likefun, x0 = pars, method = 'L-BFGS-B', jac = gradfun, options = {'maxiter': 1000000, 'ftol':1e-30, 'gtol':1e-4})
print res
resstr = '#' + '\t'.join([str(el) for el in res.x])
print resstr + '\n',

print res.x[-num_pf_params:]
