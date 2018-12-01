from __future__ import division, print_function
import argparse
import tensorflow as tf
import numpy as np
import pysam
from locuscollectors import NonCandidateCollector, CandidateCollector
import util as ut
import cyprocessreads as cpr
import cyregression as cre
import cygetcolumndata as cgcd
import h5py
import h5py_util

import beta_with_spikes_integrated as bws
import retry_tensorflow_neural_net as rt

import numpy.random as npr

# just for testing
import deepdish as dd


def complement(bases):
    return ''.join(['TGCA'['ACGT'.index(base)] for base in bases])

def get_context_data(
        cons,
        position,
        context_len,
        circular=True,
        onehot=False,
        max_context_size=100):
    # note: position is 1-based. converting here to 0-based to work with array
    position -= 1
    conslen = len(cons)
    if position < 0 or position >= conslen:
        raise ValueError('cannot get context for position {}: invalid position'.format(
            position+1))

    # check it's valid if the chromosome isn't circular
    if not circular:
        if position-context_len<0 or position+context_len >= conslen:
            raise ValueError("cannot get context for position {} "
                             "with reference length {}".format(position,conslen))

    if circular:
        cons = cons + cons + cons  # 1) add the same sequence before and after
        position += conslen        # 2) amend position to reflect new sequence (note conslen not used again)

    forward_context_bases = cons[position-context_len:position]
    #reverse_context_bases = cgcd.complement(cons[position:position-context_len:-1])   # still forward translation
    reverse_context_bases = complement(cons[position+2:position:-1])   # still forward translation
    if onehot:
        raise NotImplementedError('onehot not yet implemented')
    else:
        forward_context = np.zeros(4*context_len, dtype = np.float32)
        for i in xrange(context_len):
            forward_context[i*4+'ACGT'.index(forward_context_bases[i])] = 1
        reverse_context = np.zeros(4*context_len, dtype = np.float32)
        for i in xrange(context_len):
            reverse_context[i*4+'ACGT'.index(reverse_context_bases[i])] = 1

    return forward_context, reverse_context



def get_contamination(position, position_consensuses):
    forward_bases, forward_counts = np.unique(position_consensuses, return_counts=True)
    forward_fracs = forward_counts.astype(np.float64)/len(position_consensuses)
    forward_contam = np.zeros(4, dtype = np.float32)
    for base, frac in zip(forward_bases, forward_fracs):
        forward_contam['ACGT'.index(base)] = frac
    reverse_contam = forward_contam[::-1].copy()
    return forward_contam, reverse_contam


def get_bam_data(bam_fn, all_bam_fns):
    bam_data = np.zeros(len(all_bam_fns), dtype = np.float32)
    bam_data[all_bam_fns.index(bam_fn)] = 1.0
    return bam_data




desc = 'jointly infer sequencing error profiles and polymorphisms'
parser = argparse.ArgumentParser(
        description=desc,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('bams', help = 'file containing list of bam files')
parser.add_argument('references',
        help = 'name of file containing list of reference sequence names, or '
               'comma-separated list of reference sequence names')
parser.add_argument('numepochs', help = 'number of training epochs', type = int)
parser.add_argument('--min-bq', help = 'minimum base quality',
        type = ut.positive_int, default = 20)
parser.add_argument('--min-mq', help = 'minimum mapping quality',
        type = ut.positive_int, default = 20)
parser.add_argument('--bam-data',
        help = 'comma-separated (csv) dataset containing supplementary '
               'variables for different bam files. added to covariates for '
               'each observation from a bam file. first column is the bam '
               'name, remaining columns are the covariates. must be '
               'dummy-encoded.')
parser.add_argument('--context-length', type = ut.nonneg_int, default = 2,
        help = 'number of preceding bases to use as covariate')
parser.add_argument('--round-distance-by',
        type = ut.positive_int, default = 1,
        help = 'round distance from start of read by this amount. larger '
               'numbers make for more compression in the data, faster '
               'likelihood evaluations.')
parser.add_argument('--no-mapq', action = 'store_true',
        help = 'do not use map qualities')
parser.add_argument('--no-bam', action = 'store_true',
        help = 'do not add dummy variable for bam')
parser.add_argument('--num-hidden-layers', type = int, nargs = '+', default = [50],
        help = 'hidden layer sizes, space-delimited')
parser.add_argument('--min-coverage', type = int, default = 20)
parser.add_argument('--do-not-remove-nonvariable', action = 'store_true')
args = parser.parse_args()
min_bq = args.min_bq
min_mq = args.min_mq
context_len = args.context_length

# break bam names into prefix/name
prefix, bam_fns, bams = ut.get_bams(args.bams)
ref_names = ut.get_ref_names(args.references, bams)

num_bams = len(bam_fns)

print('getting counts')
all_counts = ut.get_all_counts(bams, ref_names, min_bq)
print('getting consensus')
all_consensuses = ut.get_all_consensuses(all_counts, min_coverage = args.min_coverage)
print('getting major-minor')
all_majorminor = ut.get_all_majorminor(all_counts)

savedat = (all_counts, all_consensuses, all_majorminor)
dd.io.save('debug_data.h5', savedat)

##################################################################################
# for debugging only
###################################
#print('(loading data from deepdish for prototyping purposes)')
#all_counts, all_consensuses, all_majorminor = dd.io.load('debug_data.h5')

nfreqs = 200
freqs = bws.get_freqs(nfreqs)
windows = bws.get_window_boundaries(nfreqs)
logf = np.log(freqs)
log1mf = np.log(1-freqs)

ncols = 4
ncols_const = 12 + len(bam_fns)
hidden_layer_sizes = [20,20]

# TODO figure out ncols and ncols_const ahead of time
ll_aux = rt.get_ll_gradient_and_inputs(ncols, ncols_const, hidden_layer_sizes, nfreqs)
init = tf.global_variables_initializer()

num_pf_params = 3

nparams = ll_aux[2].shape[0] + num_pf_params
init_params = npr.normal(size = nparams)/10000


import time

example_positions = range(900,1000, 2)



##################################################################################

all_loci = []
for ref in ref_names:
    for bamfn in bams.keys():
        cons = all_consensuses[ref][bamfn]
        reflen = len(cons)
        for i in range(1,reflen+1):
            all_loci.append((ref, bamfn, i))
all_loci = np.array(all_loci)


sess = tf.Session()
sess.run(init)


def get_data(dataslice):
    ref, bamfn, position = dataslice
    bam = bams[bamfn]
    ref = bytes(ref)
    position = int(position)
    position_consensuses = [all_consensuses[ref][bamp][position] for bamp in all_consensuses[ref].keys()]
    cons = all_consensuses[ref][bamfn]
    major, minor = all_majorminor[ref][bamfn][position]
    reflen = len(cons)
    start = time.time()
    forward_data, reverse_data = cgcd.get_column_data(
            bam,
            ref,
            reflen,
            position,
            args.min_bq,
            args.min_mq,
            args.context_length,
            bytes(cons),
            args.round_distance_by
            )
    dur = time.time()-start
    for_cov = forward_data.covariate_matrix().copy().astype(np.float32)
    for_obs = forward_data.observations().copy().astype(np.float32)
    rev_cov = reverse_data.covariate_matrix().copy().astype(np.float32)
    rev_obs = reverse_data.observations().copy().astype(np.float32)

    forward_context, reverse_context = get_context_data(cons,position, args.context_length)
    forward_contam, reverse_contam = get_contamination(position, position_consensuses)

    bam_data = get_bam_data(bamfn, bam_fns)

    forward_const_cov = np.concatenate((forward_context, forward_contam, bam_data))
    reverse_const_cov = np.concatenate((reverse_context, reverse_contam, bam_data))
    return (for_cov, for_obs, rev_cov, rev_obs, forward_context,
            reverse_context, forward_contam, reverse_contam, forward_const_cov,
            reverse_const_cov, major, minor, position)


all_loci_tf = tf.constant(all_loci)
ret_type = tuple([tf.float32]*10 + [tf.string]*2 + [tf.int64])

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for epoch in xrange(args.numepochs):
    all_loci_shuf_tf = tf.constant(npr.permutation(all_loci))
    dataset = tf.data.Dataset.from_tensor_slices(all_loci_shuf_tf).map(lambda x: tf.py_func(get_data, [x], ret_type, stateful = False))
    dataset = dataset.shuffle(1, reshuffle_each_iteration=True).batch(1)
    it = dataset.make_initializable_iterator()
    next_data = it.get_next()
    sess.run(it.initializer)
    while True:
        try:
            (for_cov, for_obs, rev_cov, rev_obs, forward_context,
                    reverse_context, forward_contam, reverse_contam, forward_const_cov,
                    reverse_const_cov, major, minor, position) = sess.run(next_data)
        except tf.errors.OutOfRangeError:
            break
        major = major[0]
        minor = minor[0]

        start = time.time()
        ll, grads = rt.loglike_and_gradient_wrapper(init_params, for_cov[0],
                rev_cov[0], forward_const_cov[0], reverse_const_cov[0], for_obs[0],
                rev_obs[0], major, minor, num_pf_params, logf, log1mf, freqs,
                windows, ll_aux, sess)
        dur = time.time()-start
        print('took {} seconds for position {}'.format(dur, position[0]))
