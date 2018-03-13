import deepdish as dd
import numpy as np
import numpy.random as npr
import afd
import gradient
import cygradient
import cylikelihood
import beta_with_spikes as bws
import util

num_f = 100
f = bws.get_freqs(num_f)
lf = np.log(f)
l1mf = np.log(1-f)

cm, lo, all_majorminor = dd.io.load('empirical_onecm.h5')
cm, cm_minmaxes = util.normalize_covariates(cm)

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
npr.seed(0); betas = npr.uniform(-0.2,0.2, size=nbetas)
num_pf_params = 3
a, b, z = -1, 0.5, -0.5
#a, b, z = 0,0,0
#a, b, z = -1, -1, -2
pars = np.concatenate((betas, (a,b,z)))

if __name__ == '__main__':
    bam = bam_fns[0]
    loc = 0

    import schwimmbad
    pool = schwimmbad.MultiPool(4)

    #grad = cygradient.gradient(pars, cm, lo, all_majorminor, blims, rowlen,
    #        f, lf, l1mf, regkeys, num_f=100,num_pf_params=3,pool=pool)
    #for g in grad:
    #    print g
    #ll = cylikelihood.ll(pars, cm, lo, all_majorminor, blims, rowlen, f, lf,
    #        l1mf, regkeys, num_f=100, num_pf_params=3, pool=pool)
    #print ll

    grad = cygradient.loc_gradient_make_buffers(pars, 'chrM', bam, loc, cm, lo, all_majorminor, blims, rowlen,
            f, lf, l1mf, regkeys, num_f=100,num_pf_params=3)
    for g in grad:
        print g

    #likefun = lambda x: gradient.grad_locus_log_likelihood(x, 'chrM', bam, loc, cm, lo, all_majorminor, blims, rowlen, f, lf, l1mf, regkeys, num_f=100,num_pf_params=3)
    #from scipy.optimize import approx_fprime
    #ngrad = approx_fprime(pars, likefun, 1e-9)
    #for ng in ngrad:
    #    print ng

    #import line_profiler
    #prof = line_profiler.LineProfiler(cygradient.collect_alpha_delta_log_summands)
    #prof.runcall(cygradient.loc_gradient_make_buffers, pars, 'chrM', bam, loc, cm, lo, all_majorminor, blims, rowlen, f, lf, l1mf, regkeys, len(f), 3)
    #prof.print_stats()

    #def loc_gradient_make_buffers(params, ref, bam, position, cm, lo, mm, blims,
    #        rowlen, freqs, lf, l1mf, regs, num_f, num_pf_params):

    #import cProfile
    #cProfile.runctx("cygradient.loc_gradient_make_buffers(pars, 'chrM', bam, loc, cm, lo, all_majorminor, blims, rowlen, f, lf, l1mf, regkeys, len(f), 3)", globals(), locals(), "Profile.prof")
    #import pstats
    #s = pstats.Stats("Profile.prof")
    #s.strip_dirs().sort_stats("time").print_stats()



    #from likelihood import single_locus_log_likelihood
    #likefun = lambda x: single_locus_log_likelihood(x, 'chrM', bam, loc, cm, lo, all_majorminor, blims, rowlen, freqs, breaks, lf, l1mf, regkeys)
    #ll_ll = single_locus_log_likelihood(pars, 'chrM', bam, loc, cm, lo, all_majorminor, blims, rowlen, freqs, breaks, lf, l1mf, regkeys)
    #ll_grad = gradient.grad_locus_log_likelihood(pars, 'chrM', bam, loc, cm, lo, all_majorminor, blims, rowlen, freqs, breaks, lf, l1mf, regkeys)
    #print ll_grad, ll_ll, ll_grad/ll_ll, ll_grad - ll_ll
