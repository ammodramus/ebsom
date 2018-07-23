import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import scipy.optimize as opt
import h5py
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
import datetime
from scipy.special import logsumexp
import cyutil as cut
import numpy.random as npr

import h5py_util

parser = argparse.ArgumentParser(
        description='split error modeling files into training and target',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('input', help = 'input HDF5 file')
parser.add_argument('params',
        help = 'file(s) with parameters, one per line. do not include allele-frequency distribution parameters',
        nargs = '+')
parser.add_argument('--num-loci', type = int, help = 'randomly sample this many loci from each bam')
args = parser.parse_args()

print '#' + ' '.join(sys.argv)

dat = h5py.File(args.input, 'r')
print '# loading all_majorminor'
all_majorminor = h5py_util.get_major_minor(dat)
print '# obtaining column names'
colnames_str = dat.attrs['covariate_column_names']
colnames = colnames_str.split(',')

h5cm = dat['covariate_matrices']
h5lo = dat['locus_observations']
h5mm = dat['major_minor']

regkeys = [(b, r) for b in 'ACGT' for r in (1,2)]
rowlen = dat.attrs['rowlen']
blims = {}
for i, reg in enumerate(regkeys):
    low = rowlen*3*i
    high = rowlen*3*(i+1)
    blims[reg] = (low, high)

nbetas = len(regkeys)*3*rowlen
pars = []
for parfile in args.params:
    pars.append(np.loadtxt(parfile))

num_pf_params = 3
betas = []
for parset in pars:
    betas.append(parset[:-num_pf_params])

column_names = []
column_names.extend(['direction', 'major', 'minor', 'rmajor', 'rminor', 'chrom', 'bam', 'locus'])
for param_idx, param in enumerate(args.params):
    for regkey in regkeys:
        regrep = str(regkey[0]) + str(regkey[1])
        for outcome in 'ACGT':
            col = '_'.join([str(param_idx), regrep, outcome])
            column_names.append(col)
print '\t'.join(column_names)

# TODO write header with columns

for chrom in h5cm.keys():
    for bam in h5cm[chrom].keys():
        tmm = h5mm[chrom][bam][:]
        if args.num_loci and args.num_loci < len(h5cm[chrom][bam].keys()):
            sampled_loci = npr.choice(h5cm[chrom][bam].keys(), size = args.num_loci, replace = False)
        else:
            sampled_loci = h5cm[chrom][bam].keys()
        for locus in sampled_loci:
            tcm = h5cm[chrom][bam][locus][:]
            X = tcm
            logprobs = []
            for betaset in betas:
                tlogprobs = {}
                for reg in regkeys:
                    low, high = blims[reg]
                    b = betaset[low:high].reshape((rowlen,-1), order = 'F')
                    Xb = np.column_stack((np.dot(X,b), np.zeros(X.shape[0])))
                    Xb -= logsumexp(Xb, axis = 1)[:,None]
                    tlogprobs[reg] = Xb
                logprobs.append(tlogprobs)
            loc_int = int(locus)
            major, minor = tmm[loc_int,:]
            major, minor = str(major), str(minor)
            rmajor, rminor = cut.comp(major), cut.comp(minor)
            majorminors = [major, minor, rmajor, rminor]
            context = [chrom, bam, locus]
            for direc in 'fr':
                for rn in '12':
                    dr = direc + rn
                    tlo = h5lo[chrom][bam][locus][dr][:]
                    if 0 in tlo.shape:
                        continue
                    for obsidx in tlo[:,0]:
                        tlp = []
                        for tlogprobs in logprobs:
                            for reg in regkeys:
                                tlp.extend(list(tlogprobs[reg][obsidx,:]))
                        res = [str(el) for el in [dr] + majorminors + context + tlp]
                        print '\t'.join(res)
