from __future__ import division, print_function
import re
import argparse
from itertools import izip
import os.path
import glob

import numpy as np
import numpy.random as npr
import tensorflow as tf
import tables
from scipy.special import logsumexp

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
parser.add_argument('--num-freqs', type=int, default=100,
                    help='number of discrete allele frequencies')
parser.add_argument('--concentration-factor', type=float, default=10,
                    help='increase to increase concentration of frequencies '
                         'near zero')
parser.add_argument('--batch-size', type=int, default=100,
                    help='number of base-pair positions to process for each '
                         'batch')
parser.add_argument('--learning-rate', type=float, default=0.01)
parser.add_argument('--checkpoint-prefix', type=str,
                    help='save model parameters to this file prefix')
parser.add_argument('--checkpoint-interval', type=int, default=100,
                    help='interval of batches between parameter checkpointing')
parser.add_argument('--restore-prefix', type=str,
                    help='restore from tensorflow checkpoint'
                         ' (must have same --hidden-layer-sizes and data)')
args = parser.parse_args()

do_checkpoint = (args.checkpoint_prefix is not None)

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

opt = tf.train.RMSPropOptimizer(args.learning_rate)
global_step = tf.train.get_or_create_global_step()

freqs = bws.get_freqs(args.num_freqs)
windows = bws.get_freqs(args.num_freqs)

try:
    dat = tables.File(args.data)
    err_mod = errormodel.ErrorModel(dat, args.hidden_layer_sizes,
                                    args.num_freqs, args.concentration_factor,
                                    num_pf_params, sess)
    params_tf = err_mod.ll_aux[2]
    grads_tf = tf.placeholder(shape=params_tf.shape, dtype=tf.float32)
    apply_grads = opt.apply_gradients([(grads_tf, params_tf)], global_step)

    # register summaries
    with tf.name_scope('pf_pars'):
        pf_param_0_tf = tf.summary.scalar('pfpar0', params_tf[0])
        pf_param_1_tf = tf.summary.scalar('pfpar1', params_tf[1])
        pf_param_2_tf = tf.summary.scalar('pfpar2', params_tf[2])
        probnon0_tf = tf.placeholder(shape=(), dtype=tf.float32)
        probnon0_summary = tf.summary.scalar('probnonzero', probnon0_tf)
    with tf.name_scope('pf_par_grads'):
        pf_param_0_grad_tf = tf.summary.scalar('pfpar0grad', grads_tf[0])
        pf_param_1_grad_tf = tf.summary.scalar('pfpar1grad', grads_tf[1])
        pf_param_2_grad_tf = tf.summary.scalar('pfpar2grad', grads_tf[2])
    with tf.name_scope('ll'):
        tot_ll_tf = tf.placeholder(shape=(), dtype=tf.float32)
        tot_ll_tf_summary = tf.summary.scalar('tot_ll', tot_ll_tf)
    with tf.name_scope('param_moments'):
        nn_params_tf = params_tf[num_pf_params:]
        param_mean_tf = tf.reduce_mean(nn_params_tf)
        mean_summary = tf.summary.scalar('m1', param_mean_tf)
        param_m2_tf = tf.reduce_mean((nn_params_tf - param_mean_tf)**2)
        var_summary = tf.summary.scalar('m2', param_m2_tf)
        param_m3_tf = tf.reduce_mean((nn_params_tf - param_mean_tf)**3)
        m3_summary = tf.summary.scalar('m3', param_m3_tf)
        param_m4_tf = tf.reduce_mean((nn_params_tf - param_mean_tf)**4)
        m4_summary = tf.summary.scalar('m4', param_m4_tf)
    with tf.name_scope('pfprobhistogram'):
        n_sample = 1000
        freqsample_tf = tf.placeholder(shape=(n_sample,), dtype=tf.float32)
        freqsample_summ = tf.summary.histogram('freqsample', freqsample_tf)

    with tf.name_scope('saver'):
        saver = tf.train.Saver()


    summaries_tf = tf.summary.merge_all()

    with tf.name_scope('filewriter'):
        prev_runs = glob.glob('summaries/*')
        if len(prev_runs) == 0:
            this_run = 1
        else:
            prev_run_idx = max(
                [int(rundir.split('n')[1]) for rundir in prev_runs])
            this_run = prev_run_idx + 1
        this_rundir = 'summaries/run{:03d}'.format(this_run)
        writer = tf.summary.FileWriter(this_rundir, sess.graph)


    with tf.name_scope('progress_bar'):
        progbar = tf.keras.utils.Progbar(args.numepochs
                                         * err_mod.metadata_np.shape[0])



    sess.run(tf.global_variables_initializer())

    if args.restore_prefix is not None:
        saver.restore(sess, args.restore_prefix)


    global_batch_idx = 0
    for epoch in xrange(args.numepochs):
        gen = lambda: bam_ref_pos_generator(err_mod)

        dataset = tf.data.Dataset.from_generator(
            gen, ret_type).batch(args.batch_size)
        it = dataset.make_initializable_iterator()
        next_data = it.get_next()
        sess.run(it.initializer)
        while True:
            progbar.update(global_batch_idx)
            try:
                batch_bams, batch_refs, batch_positions = sess.run(next_data)
            except tf.errors.OutOfRangeError:
                break

            tot_ll = 0.0
            tot_grads_minimize = 0.0
            for bam, ref, pos in izip(batch_bams, batch_refs, batch_positions):
                ll, grads = err_mod.loglike_and_gradient(bam, ref, pos)
                grads_minimize = -1.0*grads
                tot_ll += ll
                tot_grads_minimize += grads_minimize
            # normalize gradient by batch size
            tot_grads_minimize /= args.batch_size
            ll_per_locus = tot_ll / args.batch_size
            sess.run(apply_grads, feed_dict={grads_tf:tot_grads_minimize})

            pf_params = sess.run(params_tf[:num_pf_params])
            lpf_np = bws.get_lpf(pf_params, freqs, windows)
            probnonzero = 1-np.exp(lpf_np[0])
            distn = np.exp(lpf_np)
            distn /= distn.sum()
            freqsample_np = npr.choice(freqs, size=n_sample, p=distn,
                                       replace=True)
            feed_dict = {grads_tf:grads_minimize, tot_ll_tf:ll_per_locus,
                         probnon0_tf:probnonzero, freqsample_tf:freqsample_np}
            summ = sess.run(summaries_tf, feed_dict=feed_dict)
            writer.add_summary(summ, global_batch_idx)

            global_batch_idx += 1
            if do_checkpoint and (global_batch_idx % args.checkpoint_interval
                                  == 0):
                saver.save(sess, './'+args.checkpoint_prefix)

finally:
    dat.close()
