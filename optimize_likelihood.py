from __future__ import print_function
import schwimmbad
import scipy.optimize as opt
import deepdish as dd
import numpy as np
import afd
from likelihood import calc_likelihood
import numpy.random as npr

N = 1000
ab = 0.3
ppoly = 0.1
breaks = afd.get_breaks_symmetric(N = N, uniform_weight = 0.2, min_bin_size = 0.01)
freqs = afd.get_binned_frequencies(N, breaks)
lf = np.log(freqs)
l1mf = np.log(1-freqs)

cm, lo, all_majorminor = dd.io.load('empirical_onecm.h5')

key = lo['chrM'].keys()[0]
lo_sorted = {}
for chrom, lochrom in lo.iteritems():
    lo_sorted[chrom] = {}
    for bam_fn, lobam in lochrom.iteritems():
        lo_sorted[chrom][bam_fn] = []
        for locidx, loclo in enumerate(lobam):
            thislo = []
            for fr in [0,1]:
                p = []
                for r in [0,1]:
                    a = loclo[fr][r]
                    p.append(a[np.argsort(a[:,0])].astype(np.uint32).copy())
                p = tuple(p)
                thislo.append(p)
            thislo = tuple(thislo)
            lo_sorted[chrom][bam_fn].append(thislo)

regkeys = [(b, r) for b in 'ACGT' for r in (1,2)]
rowlen = cm.shape[1]
blims = {}
for i, reg in enumerate(regkeys):
    low = rowlen*3*i
    high = rowlen*3*(i+1)
    blims[reg] = (low, high)

nbetas = len(regkeys) * 3 * rowlen

parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--swarm', action = 'store_true')
parser.add_argument('--mpi', action = 'store_true')
parser.add_argument('--num-processes', '-n', type = int, default = 10)
args = parser.parse_args()

target = lambda x: -1*calc_likelihood(x, cm, lo, all_majorminor, blims, rowlen,
        freqs, breaks, lf, l1mf, regkeys, pool, printres = True)

if not args.mpi:
    pool = schwimmbad.MultiPool(args.num_processes)
else:
    pool = schwimmbad.MPIPool()
    if not pool.is_master():
        pool.wait()
        sys.exit(0)

if args.swarm:
    from pyswarm import pso
    lbbeta = -15
    ubbeta = 15
    lbab = 1e-8
    ubab = 10
    lbppoly = 1e-8
    ubppoly = 0.1
    lb = [lbbeta]*nbetas + [lbab,lbppoly]
    ub = [ubbeta]*nbetas + [ubab,ubppoly]
    xopt, fopt = pso(target, lb, ub)

else:
    betas = npr.uniform(-0.1, 0.1, nbetas)
    pars = np.concatenate((betas, (ab, ppoly)))
    opts = {'maxiter': 100000}
    res = opt.minimize(target, pars, method = 'Nelder-Mead', options = opts)
    print(res)
