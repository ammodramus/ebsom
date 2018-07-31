from __future__ import division
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


parser = argparse.ArgumentParser(
        description='optimize neural-net-based model of heteroplasmy \
                and sequencing error',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('input', help = 'input HDF5 file')
parser.add_argument('gradientL1norm', help = 'L1 norm, for gradient of distribution or NN parameters, to be reached before switching optimization target')
parser.add_argument('--bad-locus-file')
parser.add_argument('--init-params',
        help = 'file with initial parameters')
parser.add_argument('--num-epochs', type = int, default = 100)
parser.add_argument('--batch-size', type = int, default = 32)
parser.add_argument('--distribution-alpha', help = 'learning rate for distribution parameters only', type = float)
parser.add_argument('--alpha', type = float, default = 0.01, help = 'learning rate after polymorphism is introduced')
parser.add_argument('--num-hidden-layers', type = int, nargs = '+', default = [50],
        help = 'hidden layer sizes, space-delimited')
parser.add_argument('--dropout-keep-prob', type = float, default = 1.0, help = 'dropout keep probability for training')
parser.add_argument('--num-frequencies', default = 100, help = 'number of discrete frequencies to model', type = int)
parser.add_argument('--concentration-factor', default = 10, help = '"concentration factor" for frequency spacing. Defaults to 10, equal to PSMC spacing', type = int)
args = parser.parse_args()

num_f = args.num_frequencies
conc_factor = args.concentration_factor
freqs = bws.get_freqs(num_f, conc_factor)
windows = bws.get_window_boundaries(num_f, conc_factor)
lf = np.log(freqs)
l1mf = np.log(1-freqs)


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
    a, b, z = -1, 0.5, 8
    # note: pf_params come first
    pars = np.concatenate(((a,b,z), betas))

ll_aux = tfnn.get_ll_and_grads_tf(num_inputs, hidden_layer_sizes, num_f)

init = tf.global_variables_initializer()
session = tf.Session()
session.run(init)

arglist = get_args(good_keys, all_majorminor)  # each element is (key, major, minor)

num_args = len(arglist)
bs = args.init_batch_size if args.init_batch_size is not None else args.batch_size
W = pars.copy()

alpha = np.ones_like(W)*args.alpha
if args.distribution_alpha:
    alpha[:num_pf_params] = args.distribution_alpha

# these are the args in the call to grad_target, following batch
remaining_args = [num_pf_params, lf, l1mf, freqs, windows, ll_aux, session]

def grad_target(params, batch, num_pf_params, logf, log1mf, freqs, windows, ll_aux, session):
    loc_gradient_args = []
    lls = []
    grads = []
    total_count = 0
    for key, major, minor in batch:
        cm = h5cm[key][:]
        lo = h5lo[key]
        lo = [[lo['f1'][:], lo['f2'][:]], [lo['r1'][:], lo['r2'][:]]]
        #total_count += lo[0][0][:,1:].sum() + lo[0][1][:,1:].sum() + lo[1][0][:,1:].sum() + lo[1][1][:,1:].sum()
        for i in range(2):
            for j in range(2):
                if 0 in lo[i][j].shape:
                    continue
                total_count += lo[i][j][:,1:].sum()
        num_obs = cm.shape[0]
        ll, grad = tfnn.loglike_and_gradient_wrapper(params, cm, lo, major,
                minor, num_pf_params, logf, log1mf, freqs, windows, args.dropout_keep_prob, ll_aux,
                session)
        lls.append(ll)
        grads.append(grad)
    # Divide grad also by total number of loci, since each locus can contain an
    # enormous amount of data. Without doing this, huge steps based on a single
    # batch. Divide by 1000.0 because most loci will not count much in the
    # likelihood once the error model is learned. Also because it seems to
    # improve things.
    grad = np.mean(grads, axis = 0)/(total_count / len(batch) / 1000.0)
    return -1.0*grad

split_at = np.arange(0, num_args, args.batch_size)[1:]

n_completed_epochs = 0
mode = 'nn'
while True:
    permuted_args = npr.permutation(arglist)
    batches = np.array_split(permuted_args, split_at)
    for j, batch in enumerate(batches):
        Wgrad = grad_target(W, batch, *remaining_args)
        print '# grad:' + '\t'.join(['x', 'x', 'x'] + ['{:.4e}'.format(el) for el in Wgrad])
        if mode == 'nn':
            Wnorm = np.sum(np.abs(Wgrad[num_pf_params:]))
            print '## grad L1 norm (nn mode):', Wnorm
            W[num_pf_params:] += -alpha * Wgrad[num_pf_params:]
        else:
            Wnorm = np.sum(np.abs(Wgrad[:num_pf_params]))
            print '## grad L1 norm (dist mode):', Wnorm
            W[:num_pf_params] += -alpha * Wgrad[:num_pf_params]
        if Wnorm <= args.gradientL1norm:
            # Switch 'modes', i.e., optimization targets
            mode = 'dist' if mode == 'nn' else 'nn'
        ttime = str(datetime.datetime.now()).replace(' ', '_')
        print "\t".join([str(n_completed_epochs), str(j), ttime] + ['{:.4e}'.format(el) for el in W])

    n_completed_epochs += 1
    if n_completed_epochs >= args.num_epochs:
        break
