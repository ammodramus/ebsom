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

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
num_workers = comm.size-1

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
parser.add_argument('--bad-locus-file')
parser.add_argument('--init-params',
        help = 'file with initial parameters')
parser.add_argument('--mpi', action = 'store_true')
parser.add_argument('--num-processes', type = int, default = 1)
parser.add_argument('--num-reps', type = int, default = 100)
parser.add_argument('--batch-size', type = int, default = 16)
parser.add_argument('--grad-clip', help = 'max absolute value of a gradient term', type = float, default = np.inf)
parser.add_argument('--init-batch-size', type = int)
parser.add_argument('--distribution-alpha', help = 'learning rate for distribution parameters only', type = float)
parser.add_argument('--alpha', type = float, default = 0.01, help = 'learning rate after polymorphism is introduced')
parser.add_argument('--b1', type = float, help = 'decay rate for velocity mean', default = 0.9)
parser.add_argument('--b2', type = float, help = 'decay rate for acceleration mean', default = 0.999)
parser.add_argument('--init-alpha', type = float, help = 'learning rate for first stage of optimization, without heteroplasmy')
parser.add_argument('--time-with-polymorphism', type = int, default = 0, help = 'value of t in ADAM algorithm after introducing polymorphism. higher means slower learning in this phase')
parser.add_argument('--num-no-polymorphism-training-batches', '-n', type = int, default = 0,
        help = 'number of loci to consider before allowing polymoprhism')
parser.add_argument('--num-hidden-layers', type = int, nargs = '+', default = [50],
        help = 'hidden layer sizes, space-delimited')
parser.add_argument('--dropout-keep-prob', type = float, default = 1.0, help = 'dropout keep probability for training')
parser.add_argument('--num-frequencies', default = 100, help = 'number of discrete frequencies to model', type = int)
parser.add_argument('--concentration-factor', default = 10, help = '"concentration factor" for frequency spacing. Defaults to 10, equal to PSMC spacing', type = int)
parser.add_argument('--print-interval', type = int, default = 1)
parser.add_argument('--bad-keys', help = 'file containing list of keys to exclude (each is "chrom/bam/locus", where locus is 0-based)')
parser.add_argument('--gentle-release', action = 'store_true', help = "'Gently release' the probability that a site is polymorphic")
parser.add_argument('--num-batches-to-load', type = int, default = 10)
args = parser.parse_args()

if args.init_params is not None and args.num_no_polymorphism_training_batches > 0:
    raise ValueError('cannot specify both --init-params and --num-no-polymorphism-training-batches')

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

if rank != 0:
    first_batch = comm.recv(source = 0)
    print 'recv\'ed batch in {}'.format(rank)
    batch_data = []
    for key, major, minor in first_batch:
        cm = h5cm[key][:]
        lo = h5lo[key]
        lo = [[lo['f1'][:], lo['f2'][:]], [lo['r1'][:], lo['r2'][:]]]
        batch_data.append((cm, lo, major, minor))
    print 'completed first batch processing in {}'.format(rank)
    while True:
        print 'waiting for next batch in {}'.format(rank)
        batch = comm.recv(source = 0)
        print 'received next batch in {}'.format(rank)
        comm.send(batch_data, dest = 0)
        if 'exit' in batch:
            sys.exit(0)
        elif 'wait' in batch:
            print 'waiting on {}'.format(rank)
            continue
        batch_data = []
        for key, major, minor in batch:
            cm = h5cm[key][:]
            lo = h5lo[key]
            lo = [[lo['f1'][:], lo['f2'][:]], [lo['r1'][:], lo['r2'][:]]]
            data = (cm, lo, major, minor)
            batch_data.append(data)

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

if args.bad_keys is not None:
    bad_keys = set(np.loadtxt(args.bad_keys, dtype = str))
    good_keys_with_polym = good_keys  # determined up until now
    good_keys = [key for key in good_keys_with_polym if key not in bad_keys]
    print '# removed {} loci from --bad-keys file'.format(len(good_keys_with_polym)-len(good_keys))

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
bs = args.init_batch_size if args.init_batch_size is not None else args.batch_size
split_at = np.arange(0, num_args, bs)[1:]

b1 = args.b1
b2 = args.b2
eps = 1e-8
W = pars.copy()
m = 0
v = 0
t = 0

alpha = np.ones_like(W)*args.alpha
if args.distribution_alpha:
    alpha[:num_pf_params] = args.distribution_alpha

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
                minor, num_pf_params, logf, log1mf, freqs, windows, args.dropout_keep_prob, ll_aux,
                session)
        lls.append(ll)
        grads.append(grad)
    grad = np.mean(grads, axis = 0)
    return -1.0*grad

num_initial_training = 0
if (not args.init_params) and (args.num_no_polymorphism_training_batches > 0):
    ll_aux_no_poly = tfnn.get_ll_and_grads_no_poly_tf(num_inputs, hidden_layer_sizes)
    
    def grad_target_no_poly(params, batch, ll_aux, session):
        print 'entering grad_target_no_poly'
        loc_gradient_args = []
        lls = []
        grads = []
        for cm, lo, major, minor in batch:
            num_obs = cm.shape[0]
            ll, grad = tfnn.loglike_and_gradient_wrapper_no_poly(params, cm, lo, major,
                    minor, args.dropout_keep_prob, ll_aux_no_poly, session)
            lls.append(ll)
            grads.append(grad)
        grad = np.mean(grads, axis = 0)
        return -1.0*grad


    initial_pf_params = np.array((-1,0.5,20))
    W[:num_pf_params] = initial_pf_params[:]
    t = 0
    done = False

    num_f_init = 3
    freqs_init = bws.get_freqs(num_f_init, conc_factor)
    windows_init = bws.get_window_boundaries(num_f_init, conc_factor)
    lf_init = np.log(freqs_init)
    l1mf_init = np.log(1-freqs_init)
    ll_aux_init = tfnn.get_ll_and_grads_tf(num_inputs, hidden_layer_sizes, num_f_init)
    #remaining_args_init = [num_pf_params, lf_init, l1mf_init, freqs_init, windows_init, ll_aux_init, session]
    remaining_args_init = [ll_aux_init, session]

    init_batches_completed = 0
    while num_initial_training < args.num_no_polymorphism_training_batches:
        print 'entering main loop in {}'.format(rank)
        permuted_args = npr.permutation(arglist)
        batches = np.array_split(permuted_args, split_at)
        num_batches = len(batches)

        # seed the workers
        next_batch_idx = 0
        for i in range(num_workers):
            comm.send(batches[next_batch_idx], dest = i+1)
            next_batch_idx += 1
        current_worker = 0
        batches_completed = 0
        while batches_completed < num_batches:
            # have to send a batch before you get a batch... get the batch from the worker
            if next_batch_idx < num_batches:
                print 'sending next batch to {} from {}'.format(current_worker+1, rank)
                comm.send(batches[next_batch_idx], dest = current_worker+1)
                next_batch_idx += 1
            else:
                print 'sending "wait" to {} from {}'.format(current_worker+1, rank)
                comm.send('wait', dest = current_worker+1)
            print 'waiting for message from {} on {}'.format(current_worker+1, rank)
            batch = comm.recv(source = current_worker+1)
            print 'received batch message on {}, calculating gradient'.format(rank)
            current_worker = (current_worker + 1) % num_workers

            t += 1
            init_batches_completed += 1
            num_initial_training += 1
            Wgrad = grad_target_no_poly(W[num_pf_params:], batch, *remaining_args_init)
            Wgrad = np.sign(Wgrad) * np.minimum(np.abs(Wgrad), args.grad_clip)
            m = b1*m + (1-b1)*Wgrad
            v = b2*v + (1-b2)*(Wgrad*Wgrad)
            mhat = m/(1-b1**t)
            vhat = v/(1-b2**t)
            W[num_pf_params:] += -alpha[num_pf_params:] * mhat / (np.sqrt(vhat) + eps)
            ttime = str(datetime.datetime.now()).replace(' ', '_')
            batches_completed += 1
            if batches_completed % args.print_interval == 0:
                print '#' + '\t'.join(Wgrad.astype(str))
                print "\t".join([str(-1), str(init_batches_completed), ttime] + ['{:.4e}'.format(el) for el in W])
            if num_initial_training >= args.num_no_polymorphism_training_batches:
                done = True
                break

if args.bad_keys is not None:
    arglist = get_args(good_keys_with_polym, all_majorminor)  # each element is (key, major, minor)
    num_args = len(arglist)
    bs = args.init_batch_size if args.init_batch_size is not None else args.batch_size
    split_at = np.arange(0, num_args, bs)[1:]
    split_at = np.arange(0, num_args, args.batch_size)[1:]

m = 0
v = 0
t = args.time_with_polymorphism

if not args.init_params:
    if not args.gentle_release:
        post_init_pf_params = np.array((-1,0.5,7))  # corresponding to 0.9990889 prob of being fixed
        W[:num_pf_params] = post_init_pf_params[:]
    else:
        # if --gentle-release is specified, the program goes into the second
        # stage of optimization with the same distribution parameters that it
        # ended the first stage with, namely, that there is no polymorphism
        pass

n_completed_epochs = 0
while n_completed_epochs < args.num_reps:
    permuted_args = npr.permutation(arglist)
    batches = np.array_split(permuted_args, split_at)
    for j, batch in enumerate(batches):
        t += 1
        Wgrad = grad_target(W, batch, *remaining_args)
        # Apply gradient clipping
        Wgrad = np.sign(Wgrad) * np.minimum(np.abs(Wgrad), args.grad_clip)
        m = b1*m + (1-b1)*Wgrad
        v = b2*v + (1-b2)*(Wgrad*Wgrad)
        mhat = m/(1-b1**t)
        vhat = v/(1-b2**t)
        W += -alpha * mhat / (np.sqrt(vhat) + eps)
        ttime = str(datetime.datetime.now()).replace(' ', '_')
        if j % args.print_interval == 0:
            print '#x\tx\tx\t' + '\t'.join(Wgrad.astype(str))
            print "\t".join([str(n_completed_epochs), str(j), ttime] + ['{:.4e}'.format(el) for el in W])

    n_completed_epochs += 1
    if n_completed_epochs >= args.num_reps:
        break
