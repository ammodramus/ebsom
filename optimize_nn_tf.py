import h5py
import numpy as np
import numpy.random as npr
import beta_with_spikes_integrated as bws
import sys
import argparse
import datetime

import h5py_util
import neural_net as nn

import tensorflow_neural_net as tfnn
import tensorflow as tf


def get_args(locus_keys, mm):
    # args will be key, major, minor
    args = []
    for key in locus_keys:
        chrom, bam, locus = key.split('/')
        locus = int(locus)
        major, minor = mm[chrom][bam][locus]
        if major == 'N':
            continue
        args.append([key, major, minor])
    return args

def get_major_minor(h5in):
    mm = {}
    for chrom in h5in['major_minor'].keys():
        h5_chrom_mm = h5in['major_minor'][chrom]
        mm[chrom] = {}
        for bam in h5_chrom_mm.keys():
            h5_bam_mm = h5_chrom_mm[bam]
            t_h5_bam_mm = h5_bam_mm[:,:].copy()
            mm[chrom][bam] = t_h5_bam_mm
    return mm




print '#' + ' '.join(sys.argv)

num_f = 100
freqs = bws.get_freqs(num_f)
windows = bws.get_window_boundaries(num_f)
lf = np.log(freqs)
l1mf = np.log(1-freqs)

parser = argparse.ArgumentParser(
        description='optimize neural-net-based model of heteroplasmy \
                and sequencing error',
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
parser.add_argument('--num-no-polymorphism-training-batches', '-n', type = int, default = 0,
        help = 'number of loci to consider before allowing polymoprhism')
parser.add_argument('--num-hidden-layers', type = int, nargs = '+', default = [50],
        help = 'hidden layer sizes, space-delimited')
args = parser.parse_args()


dat = h5py.File(args.input, 'r')
print '# loading all_majorminor'
all_majorminor = get_major_minor(dat)
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

locus_keys = h5cm_keys
    
# badloci.txt is 1-based, these will be 0-based. notice chrM is hard-coded
if args.bad_locus_file is not None:
    badloci = set(list(np.loadtxt(args.bad_locus_file).astype(np.int)-1))
    good_keys = []
    for key in locus_keys:
        locus = int(key.split('/')[-1])
        if locus not in badloci:
            good_keys.append(key)
else:
    good_keys = h5cm_keys



num_pf_params = 3

cm_key = good_keys[0]
cm = h5cm[cm_key][:]
lo = h5lo[cm_key]
lo = [[lo['f1'][:], lo['f2'][:]], [lo['r1'][:], lo['r2'][:]]]
maj, mino = 'G', 'T'
# num_obs changes for each locus!
num_obs = cm.shape[0]
x, _, __ = nn.get_major_minor_cm_and_los(cm,lo, maj, 'T')
num_inputs = x.shape[1]
hidden_layer_sizes = args.num_hidden_layers

_, num_params = nn.set_up_neural_net(num_inputs, hidden_layer_sizes, num_obs)

if args.init_params is not None:
    pars = np.loadtxt(args.init_params)
else:
    betas = npr.uniform(-0.2, 0.2, size = num_params)
    num_pf_params = 3
    a, b, z = -1, 0.5, 8
    # note: pf_params come first
    pars = np.concatenate(((a,b,z), betas))

ll_aux = tfnn.get_ll_and_grads_tf(num_inputs, hidden_layer_sizes, num_f)

total_num_params = int(ll_aux[0].shape[0])
init = tf.global_variables_initializer()
session = tf.Session()
session.run(init)


arglist = get_args(good_keys, all_majorminor)  # each element is (key, major, minor)

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

# these are the args in the call to grad_target, following batch
remaining_args = [num_pf_params, lf, l1mf, freqs, windows, ll_aux, session]

def grad_target(params, batch, num_pf_params, logf, log1mf, freqs, windows, ll_aux, session):
    loc_gradient_args = []
    lls = []
    grads = []
    for key, major, minor in batch:
        cm = h5cm[key][:]
        lo = h5lo[key]
        lo = [[lo['f1'][:], lo['f2'][:]], [lo['r1'][:], lo['r2'][:]]]
        num_obs = cm.shape[0]
        ll, grad = tfnn.loglike_and_gradient_wrapper(params, cm, lo, major,
                minor, num_pf_params, logf, log1mf, freqs, windows, ll_aux,
                session)
        lls.append(ll)
        grads.append(grad)
    grad = np.sum(grads, axis = 0)
    return -1.0*grad

num_initial_training = 0
initial_pf_params = np.array((-1,0.5,20))
W[:num_pf_params] = initial_pf_params[:]
t = 0
done = False
while num_initial_training < args.num_no_polymorphism_training_batches:
    permuted_args = npr.permutation(arglist)
    batches = np.array_split(permuted_args, split_at)
    for j, batch in enumerate(batches):
        t += 1
        num_initial_training += 1
        Wgrad = grad_target(W, batch, *remaining_args)
        m = b1*m + (1-b1)*Wgrad
        v = b2*v + (1-b2)*(Wgrad*Wgrad)
        mhat = m/(1-b1**t)
        vhat = v/(1-b2**t)
        W += -alpha * mhat / (np.sqrt(vhat) + eps)
        # keep the probability of heteroplasmy at 1-1/(1+exp(-30))
        #W[-num_pf_params:] = initial_pf_params[:]
        W[:num_pf_params] = initial_pf_params[:]
        ttime = str(datetime.datetime.now()).replace(' ', '_')
        print "\t".join([str(-1), str(num_initial_training), ttime] + ['{:.4e}'.format(el) for el in W])
        if num_initial_training >= args.num_no_polymorphism_training_batches:
            done = True
            break

m = 0
v = 0
t = 0

post_init_pf_params = np.array((-1,0.5,7))  # corresponding to 0.9990889 prob of being fixed
W[:num_pf_params] = post_init_pf_params[:]

n_completed_reps = 0
while True:
    permuted_args = npr.permutation(arglist)
    batches = np.array_split(permuted_args, split_at)
    for j, batch in enumerate(batches):
        t += 1
        Wgrad = grad_target(W, batch, *remaining_args)
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
