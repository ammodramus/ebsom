import deepdish as dd
import numpy as np
import numpy.random as npr
import afd
import gradient
import cygradient

N = 1000
ab = 0.3
ppoly = 0.1
breaks = afd.get_breaks_symmetric(N = N, uniform_weight = 0.2, min_bin_size = 0.01)
freqs = afd.get_binned_frequencies(N, breaks)
lf = np.log(freqs)
l1mf = np.log(1-freqs)


cm, lo, all_majorminor = dd.io.load('empirical_onecm.h5')

bam_fns = lo['chrM'].keys()

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

import time; time.time()

nbetas = len(regkeys)*3*rowlen
#betas = np.zeros(nbetas)
npr.seed(0); betas = npr.uniform(-0.2,0.2, size=nbetas)
pars = np.concatenate((betas, (ab, ppoly)))

'''
import schwimmbad

pool = schwimmbad.MultiPool(4)
#pool = schwimmbad.SerialPool()
#pool = None
import time; t = time.time()
print calc_likelihood(pars, cm, lo, all_majorminor, blims, rowlen, freqs, breaks, lf,
        l1mf, regkeys, pool = pool)
dur = time.time() - t
print 'took {} seconds'.format(dur)
pool.close()
'''

if __name__ == '__main__':
    bam = bam_fns[0]
    loc = 10

    #import line_profiler
    #prof = line_profiler.LineProfiler(cygradient.gradient_make_buffers, cygradient.loc_gradient, cygradient.collect_alpha_delta_log_summands)
    #prof.runcall(cygradient.gradient_make_buffers, pars, 'chrM', bam, loc, cm, lo, all_majorminor, blims, rowlen, freqs, breaks, lf, l1mf, regkeys)
    #prof.print_stats()

    grad = cygradient.gradient_make_buffers(pars, 'chrM', bam, loc, cm, lo, all_majorminor, blims, rowlen, freqs, breaks, lf, l1mf, regkeys)
    for g in grad:
        print g
    #likefun = lambda x: gradient.grad_locus_log_likelihood(x, 'chrM', bam, loc, cm, lo, all_majorminor, blims, rowlen, freqs, breaks, lf, l1mf, regkeys)
    #print 'py ll = ', likefun(pars)

    #import cProfile
    #cProfile.runctx("cygradient.gradient_make_buffers(pars, 'chrM', bam, loc, cm, lo, all_majorminor, blims, rowlen, freqs, breaks, lf, l1mf, regkeys)", globals(), locals(), "Profile.prof")

    #import pstats
    #s = pstats.Stats("Profile.prof")
    #s.strip_dirs().sort_stats("time").print_stats()
    #grad = gradient.gradient(pars, 'chrM', bam, loc, cm, lo, all_majorminor, blims, rowlen, freqs, breaks, lf, l1mf, regkeys)
    #dur = time.time() - start
    #grad = np.concatenate((grad, (0,0)))
    #for g in grad:
    #    print g
    #print '# took {} seconds'.format(dur)
    #likefun = lambda x: gradient.grad_locus_log_likelihood(x, 'chrM', bam, loc, cm, lo, all_majorminor, blims, rowlen, freqs, breaks, lf, l1mf, regkeys)
    #from scipy.optimize import approx_fprime
    #ngrad = approx_fprime(pars, likefun, 1e-9)
    #for ng in ngrad:
    #    print ng
    #for g, ng in np.nditer((grad, ngrad)):
    #    print g, ng

    #from likelihood import single_locus_log_likelihood
    #likefun = lambda x: single_locus_log_likelihood(x, 'chrM', bam, loc, cm, lo, all_majorminor, blims, rowlen, freqs, breaks, lf, l1mf, regkeys)
    #ll_ll = single_locus_log_likelihood(pars, 'chrM', bam, loc, cm, lo, all_majorminor, blims, rowlen, freqs, breaks, lf, l1mf, regkeys)
    #ll_grad = gradient.grad_locus_log_likelihood(pars, 'chrM', bam, loc, cm, lo, all_majorminor, blims, rowlen, freqs, breaks, lf, l1mf, regkeys)
    #print ll_grad, ll_ll, ll_grad/ll_ll, ll_grad - ll_ll
