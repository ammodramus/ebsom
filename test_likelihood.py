import deepdish as dd
import numpy as np
import numpy.random as npr
import afd
from likelihood import calc_likelihood, single_locus_log_likelihood

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
betas = np.zeros(nbetas)
#npr.seed(0); betas = npr.uniform(-0.1,0.1, size=nbetas)
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
    print single_locus_log_likelihood(pars, 'chrM', bam, loc, cm, lo, all_majorminor, blims, rowlen, freqs, breaks, lf, l1mf, regkeys)
