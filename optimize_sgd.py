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

import h5py_util

print '#' + ' '.join(sys.argv)

num_f = 100
freqs = bws.get_freqs(num_f)
windows = bws.get_window_boundaries(num_f)
lf = np.log(freqs)
l1mf = np.log(1-freqs)

parser = argparse.ArgumentParser(
        description='split error modeling files into training and target',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('input', help = 'input HDF5 file')
parser.add_argument('--bad-locus-file')
parser.add_argument('--init-params',
        help = 'file with initial parameters')
parser.add_argument('--mpi', action = 'store_true')
parser.add_argument('--num-processes', type = int, default = 1)
parser.add_argument('--num-reps', type = int, default = 100)
parser.add_argument('--batch-size', type = int, default = 20)
parser.add_argument('--alpha', type = float, default = 0.01)
parser.add_argument('--restart', help = 'parameters, one per line, at which to restart optimization')
args = parser.parse_args()


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

dat = h5py.File(args.input, 'r')
print '# loading all_majorminor'
all_majorminor = h5py_util.get_major_minor(dat)
print '# obtaining column names'
colnames_str = dat.attrs['covariate_column_names']
colnames = colnames_str.split(',')

h5cm = dat['covariate_matrices']
h5lo = dat['locus_observations']

print '# getting covariate matrix keys'
h5cm_keys = []
for chrom, chrom_cm in h5cm.iteritems():
    for bam, bam_cm in chrom_cm.iteritems():
        print '# getting covariate matrix keys: {}'.format(bam)
        for locus, locus_cm in bam_cm.iteritems():
            spname = locus_cm.name.split('/')
            name = unicode('/'.join(spname[2:]))
            h5cm_keys.append(name)

print '# getting covariate matrix keys'
h5lo_keys = []
for chrom, chrom_lo in h5lo.iteritems():
    for bam, bam_lo in chrom_lo.iteritems():
        print '# getting locus observation keys: {}'.format(bam)
        for locus, locus_lo in bam_lo.iteritems():
            spname = locus_lo.name.split('/')
            name = unicode('/'.join(spname[2:]))
            h5lo_keys.append(name)
assert set(h5lo_keys) == set(h5cm_keys), "covariate matrix and locus observation keys differ"
    
bam_fns = lo['chrM'].keys()

# badloci.txt is 1-based, these will be 0-based. notice chrM is hard-coded
if args.bad_locus_file is not None:
    badloci = np.loadtxt(args.bad_locus_file).astype(np.int)-1
    for bam in bam_fns:
        for bl in badloci:
            all_majorminor['chrM'][bam][bl] = ('N', 'N')
            lo['chrM'][bam][bl] = [[[],[]], [[],[]]]

regkeys = [(b, r) for b in 'ACGT' for r in (1,2)]
rowlen = dat.attrs['rowlen']
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
    betas = npr.uniform(-0.2, 0.2, size = nbetas)
    num_pf_params = 3
    a, b, z = -1, 0.5, 8
    pars = np.concatenate((betas, (a,b,z)))



#for m, M in cm_minmaxes:
#    print '# mM {}\t{}'.format(m,M)

num_pf_params = 3

arglist = cygradient.get_args(lo, cm, all_majorminor)

num_args = len(arglist)
split_at = np.arange(0, num_args, args.batch_size)[1:]

alpha = args.alpha
b1 = 0.9
b2 = 0.999
eps = 1e-8
W = pars.copy()
m = 0
v = 0
t = 0

remaining_args = [rowlen, blims, lf, l1mf, num_pf_params, freqs,
        windows, regkeys, pool]

# TODO get the -1 right. want to maximize the likelihood
grad_target = gradient.batch_gradient_func

n_completed_reps = 0
while True:
    permuted_args = npr.permutation(arglist)
    batches = np.array_split(permuted_args, split_at)
    for j, batch in enumerate(batches):
        t += 1

        Wgrad = np.sum(grad_target(W, batch, *remaining_args), axis = 0)
        m = b1*m + (1-b1)*Wgrad
        v = b2*v + (1-b2)*(Wgrad*Wgrad)
        mhat = m/(1-b1**t)
        vhat = v/(1-b2**t)
        W += -alpha * mhat / (np.sqrt(vhat) + eps)
        ttime = str(datetime.datetime.now()).replace(' ', '_')
        print "\t".join([str(n_completed_reps), str(j), ttime] + ['{:.4e}'.format(el) for el in W])

    n_completed_reps += 1
    if n_completed_reps >= args.num_reps:
        break
