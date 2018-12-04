from __future__ import division, print_function
import re
import argparse
from itertools import izip

import numpy as np
import numpy.random as npr
import tensorflow as tf
import tables
from scipy.special import logsumexp
from tensorflow.train import GradientDescentOptimizer

import beta_with_spikes_integrated as bws
import errormodel


desc = 'jointly infer sequencing error profiles and polymorphisms'
parser = argparse.ArgumentParser(
        description=desc,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('data', help='HDF5 file of preprocessed datsa')
parser.add_argument('numepochs', help='number of training epochs', type=int)
parser.add_argument('--hidden-layer-sizes', type=int, nargs='+',
                    default=[40,40],
                    help='hidden layer sizes, space-delimited')
parser.add_argument('--num-freqs', type=int, default=100)
parser.add_argument('--concentration-factor', type=float, default=10,
                    help='increase to increase concentration of frequencies '
                         'near zero')
parser.add_argument('--batch-size', type=int, default=100,
                    help='number of base-pair positions to process for each '
                         'batch')
#parser.add_argument('--do-not-remove-nonvariable', action='store_true')
args = parser.parse_args()

num_pf_params = 3

def bam_ref_pos_generator(err_mod):
    bam_idx = err_mod.meta_cols['bam']
    ref_idx = err_mod.meta_cols['reference']
    pos_idx = err_mod.meta_cols['position']
    permut_metadat = npr.permutation(err_mod.metadata_np)
    for row in permut_metadat:
        yield (err_mod.reverse_bam_enum_values[row[bam_idx]],
               err_mod.reverse_ref_enum_values[row[ref_idx]],
               row[pos_idx])


ret_type = (tf.string, tf.string, tf.int64)

sess = tf.Session()

opt = GradientDescentOptimizer(0.001, use_locking=False)
global_step = tf.train.get_or_create_global_step()

try:
    dat = tables.File(args.data)
    err_mod = errormodel.ErrorModel(dat, args.hidden_layer_sizes,
                                    args.num_freqs, args.concentration_factor,
                                    num_pf_params, sess)

    for epoch in xrange(args.numepochs):
        gen = lambda: bam_ref_pos_generator(err_mod)

        dataset = tf.data.Dataset.from_generator(
            gen, ret_type).batch(args.batch_size)
        it = dataset.make_initializable_iterator()
        next_data = it.get_next()
        sess.run(it.initializer)
        batch_idx = 0
        while True:
            try:
                batch_bams, batch_refs, batch_positions = sess.run(next_data)
            except tf.errors.OutOfRangeError:
                break

            import time; start = time.time()
            tot_ll = 0.0
            tot_grads = 0.0
            for bam, ref, pos in izip(batch_bams, batch_refs, batch_positions):
                ll, grads = err_mod.loglike_and_gradient(params, bam, ref, pos)
                tot_ll += ll
                tot_grads += grads
            dur = time.time()-start
            print('took {} seconds for batch {}'.format(dur, batch_idx))

            #opt.apply_gradients(zip(tot_grads, 

            batch_idx += 1



except KeyboardInterrupt:
    import pdb; pdb.set_trace()
finally:
    dat.close()
