
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
parser.add_argument('--mpi', action = 'store_true')
parser.add_argument('--num-processes', type = int, default = 1)
parser.add_argument('--num-reps', type = int, default = 100)
parser.add_argument('--batch-size', type = int, default = 20)
parser.add_argument('--alpha', type = float, default = 0.01)
parser.add_argument('--restart', help = 'parameters, one per line, at which to restart optimization')
args = parser.parse_args()

dat = h5py.File(args.input, 'r')
cm = dat['covariate_matrix'][:,:]
all_majorminor = h5py_util.get_major_minor(dat)
colnames_str = dat.attrs['covariate_column_names']
colnames = colnames_str.split(',')
lo = h5py_util.get_locobs(dat, all_majorminor)

cm, cm_minmaxes = util.normalize_covariates(cm)

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
    betas = npr.uniform(-0.2, 0.2, size = nbetas)
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
    ttime = str(datetime.datetime.now()).replace(' ', '_')
    pstr = "\t".join([str(val)] + [ttime] + [str(el) for el in p])
    print pstr + '\n',
    return val

for m, M in cm_minmaxes:
    print '# mM {}\t{}'.format(m,M)

num_pf_params = 3

arglist = cygradient.get_args(lo, all_majorminor)
gradfun = cygradient.make_batch_gradient_func(cm, blims, lf, l1mf, num_pf_params, f, v, regkeys, pool)
grad_target = lambda pars, arglist: -1.0*gradfun(pars, arglist)

num_args = len(arglist)
split_at = np.arange(0, num_args, args.batch_size)[1:]
#split_at = args.batch_size

alpha = args.alpha
b1 = 0.9
b2 = 0.999
eps = 1e-8
W = pars.copy()
m = 0
v = 0
t = 0

n_completed_reps = 0
while True:
    permuted_args = npr.permutation(arglist)
    batches = np.array_split(permuted_args, split_at)
    for j, batch in enumerate(batches):
        t += 1
        Wgrad = np.sum(grad_target(W, batch), axis = 0)
        m = b1*m + (1-b1)*Wgrad
        v = b2*v + (1-b2)*(Wgrad*Wgrad)
        mhat = m/(1-b1**t)
        vhat = v/(1-b2**t)
        W += -alpha * mhat / (np.sqrt(vhat) + eps)
        ttime = str(datetime.datetime.now()).replace(' ', '_')
        print "\t".join([str(n_completed_reps), str(j), ttime] + [str(el) for el in W])

    n_completed_reps += 1
    if n_completed_reps >= args.num_reps:
        break
