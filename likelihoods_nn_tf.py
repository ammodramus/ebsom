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
parser.add_argument('params', help = 'parameter file, one parameter per line', nargs = '+')
parser.add_argument('--num-hidden-layers', type = int, nargs = '+', default = [50],
        help = 'hidden layer sizes, space-delimited')
parser.add_argument('--bad-locus-file')
parser.add_argument('--batch-size', type = int, default = 16)
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

print '# getting covariate / locus observation matrix keys'
h5cm_keys = []
for chrom, chrom_cm in h5cm.iteritems():
    for bam, bam_cm in chrom_cm.iteritems():
        print '# getting covariate matrix keys: {}'.format(bam)
        for locus, locus_cm in bam_cm.iteritems():
            spname = locus_cm.name.split('/')
            name = unicode('/'.join(spname[2:]))
            h5cm_keys.append(name)

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

all_pars = [np.loadtxt(parfile) for parfile in args.params]

ll_aux = tfnn.get_ll_and_grads_tf(num_inputs, hidden_layer_sizes, num_f)

total_num_params = int(ll_aux[0].shape[0])
init = tf.global_variables_initializer()
session = tf.Session()
session.run(init)


arglist = get_args(good_keys, all_majorminor)  # each element is (key, major, minor)

num_args = len(arglist)
bs = args.batch_size
split_at = np.arange(0, num_args, bs)[1:]

# these are the args in the call to grad_target, following batch
remaining_args = [num_pf_params, lf, l1mf, freqs, windows, ll_aux, session]

def lls_target(params, batch, num_pf_params, logf, log1mf, freqs, windows, ll_aux, session):
    loc_gradient_args = []
    lls = []
    grads = []
    for key, major, minor in batch:
        cm = h5cm[key][:]
        lo = h5lo[key]
        lo = [[lo['f1'][:], lo['f2'][:]], [lo['r1'][:], lo['r2'][:]]]
        num_obs = cm.shape[0]
        ll, grad = tfnn.loglike_and_gradient_wrapper(params, cm, lo, major,
                minor, num_pf_params, logf, log1mf, freqs, windows, 1.0, ll_aux,
                session)
        lls.append(ll)
    return lls

batches = np.array_split(arglist, split_at)
loc_num = 0
for j, batch in enumerate(batches):
    tlls = [lls_target(pars, batch, *remaining_args) for pars in all_pars]
    for i, b in enumerate(batch):
        key, major, minor = b
        chrom, bam, locus = key.split('/')
        res = [str(loc_num+i), str(num_args), chrom, bam, locus]
        res.extend([str(tlls[j][i]) for j in range(len(all_pars))])
        print '\t'.join(res)
    loc_num += len(batch)
