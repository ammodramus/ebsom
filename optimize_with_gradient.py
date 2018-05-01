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

num_f = 100
f = bws.get_freqs(num_f)
v = bws.get_window_boundaries(num_f)
lf = np.log(f)
l1mf = np.log(1-f)

parser = argparse.ArgumentParser(
        description='split error modeling files into training and target',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('input', help = 'input HDF5 file')
parser.add_argument('--bad-locus-file')
parser.add_argument('--init-params',
        help = 'file with initial parameters')
args = parser.parse_args()

dat = dd.io.load(args.input)
try:
    cm, lo, all_majorminor, colnames = dat
    have_colnames = True
except:
    cm, lo, all_majorminor = dat
    have_colnames = False

cm, cm_minmaxes = util.normalize_covariates(cm)

for m, M in cm_minmaxes:
    print '# mM {}\t{}'.format(m,M)

bam_fns = lo['chrM'].keys()

# badloci.txt is 1-based, these will be 0-based
if args.bad_locus_file is not None:
    badloci = np.loadtxt(args.bad_locus_file).astype(np.int)-1
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
#betas = np.loadtxt('global_params_1.txt')
if args.init_params is not None:
    pars = np.loadtxt(args.init_params)
else:
    import warnings
    warnings.warn('using global_params_reordered_incomplete.txt for initial parameters')
    betas = np.loadtxt('global_params_reordered_incomplete.txt')
    #betas = np.zeros(nbetas)
    num_pf_params = 3
    a, b, z = -1, 0.5, 8
    pars = np.concatenate((betas, (a,b,z)))

gradfun = lambda p: -1.0*cygradient.gradient(p, cm, lo, all_majorminor, blims, rowlen, f, v, lf, l1mf, regkeys, num_f=100,num_pf_params=3,pool=pool)

import schwimmbad
if args.mpi:
    pool = schwimmbad.MPIPool()
    if not pool.is_master():
        pool.wait()
        sys.exit(0)
else:
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


res = opt.minimize(fun = likefun, x0 = pars, method = 'L-BFGS-B', jac = gradfun, options = {'maxiter': 5000})
resstr = '#' + '\t'.join([str(el) for el in res.x])
print resstr + '\n',
