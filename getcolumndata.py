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
        onehot=False):
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

    forward_context_bases = cons[position-context_len:position]
    #reverse_context_bases = cgcd.complement(cons[position:position-context_len:-1])   # still forward translation
    reverse_context_bases = complement(cons[position+2:position:-1])   # still forward translation
    if onehot:
        raise NotImplementedError('onehot not yet implemented')
    else:
        forward_context = np.zeros(4*context_len)
        for i in xrange(context_len):
            forward_context[i*4+'ACGT'.index(forward_context_bases[i])] = 1
        reverse_context = np.zeros(4*context_len)
        for i in xrange(context_len):
            reverse_context[i*4+'ACGT'.index(reverse_context_bases[i])] = 1

    return forward_context, reverse_context



def get_contamination(position, position_consensuses):
    forward_bases, forward_counts = np.unique(position_consensuses, return_counts=True)
    forward_fracs = forward_counts.astype(np.float64)/len(position_consensuses)
    forward_contam = np.zeros(4)
    for base, frac in zip(forward_bases, forward_fracs):
        forward_contam['ACGT'.index(base)] = frac
    reverse_contam = forward_contam[::-1].copy()
    return forward_contam, reverse_contam









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

#print('getting counts')
#all_counts = ut.get_all_counts(bams, ref_names, min_bq)
#print('getting consensus')
#all_consensuses = ut.get_all_consensuses(all_counts, min_coverage = args.min_coverage)
#print('getting major-minor')
#all_majorminor = ut.get_all_majorminor(all_counts)

#savedat = (all_counts, all_consensuses, all_majorminor)
#dd.io.save('debug_data.h5', savedat)

##################################################################################
# for debugging only
###################################
print('(loading data from deepdish for prototyping purposes)')
all_counts, all_consensuses, all_majorminor = dd.io.load('debug_data.h5')

nfreqs = 200
freqs = bws.get_freqs(nfreqs)
windows = bws.get_window_boundaries(nfreqs)
logf = np.log(freqs)
log1mf = np.log(1-freqs)

ncols = 4
ncols_const = 12
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

    forward_const_cov = np.concatenate((forward_context, forward_contam))
    reverse_const_cov = np.concatenate((reverse_context, reverse_contam))
    return (for_cov, for_obs, rev_cov, rev_obs, forward_context,
            reverse_context, forward_contam, reverse_contam, forward_const_cov,
            reverse_const_cov)


all_loci_tf = tf.constant(all_loci)
ret_type = tuple([tf.float32]*10)
dataset = tf.data.Dataset.from_tensor_slices(all_loci_tf).map(lambda x: tf.py_func(get_data, [x], ret_type, stateful = False))
dataset = dataset.shuffle(all_loci_tf.shape[0], reshuffle_each_iteration=True).batch(1)
it = dataset.make_initializable_iterator()
next_data = it.get_next()

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for epoch in xrange(args.numepochs):
    sess.run(it.intializer)
    (for_cov_tf, for_obs_tf, rev_cov_tf, rev_obs_tf, forward_context_tf,
            reverse_context_tf, forward_contamination_tf, reverse_contamination_tf,
            forward_const_cov_tf, reverse_const_cov_tf) = it.get_next()

    (for_cov, for_obs, rev_cov, rev_obs, forward_context,
            reverse_context, forward_contam, reverse_contam, forward_const_cov,
            reverse_const_cov) = sess.run(next_data)

    print('getting data took {} seconds for {}'.format(dur, position))

    start = time.time()
    ll, grads = rt.loglike_and_gradient_wrapper(init_params, for_cov,
            rev_cov, forward_const_cov, reverse_const_cov, for_obs,
            rev_obs, major, minor, num_pf_params, logf, log1mf, freqs,
            windows, ll_aux, sess)
    dur = time.time()-start

    print('calculating gradient took {} seconds for {}'.format(dur, position))
