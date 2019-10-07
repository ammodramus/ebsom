from __future__ import print_function, division
import sys
import argparse
import datetime
import gc
import multiprocessing as mp
from Queue import Empty

import h5py
import numpy as np
import numpy.random as npr
import tensorflow as tf
import tensorflow.keras.layers as layers

import beta_with_spikes_integrated as bws
from likelihood_layer import Likelihood, LikelihoodLoss

MASK_VALUE = -1e28

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


def get_cm_and_lo(key, major, minor, h5cm, h5lo):
    cm = h5cm[key][:]
    lo = h5lo[key]
    cur_idx = 0
    # forward first
    lof1 = lo['f1'][:]
    lof2 = lo['f2'][:]
    # then reverse
    lor1 = lo['r1'][:]
    lor2 = lo['r2'][:]

    # add read-number column to covariates
    readnumsf1 = np.zeros(lof1.shape[0])
    readnumsf2 = np.ones(lof2.shape[0])
    readnumsf = np.concatenate((readnumsf1, readnumsf2))
    readnumsr1 = np.zeros(lor1.shape[0])
    readnumsr2 = np.ones(lor2.shape[0])
    readnumsr = np.concatenate((readnumsr1, readnumsr2))

    lof = np.vstack((lof1, lof2))
    lor = np.vstack((lor1, lor2))

    cmf = cm[lof[:,0].astype(np.int)]
    cmf = np.hstack((cmf, readnumsf[:,np.newaxis]))
    cmr = cm[lor[:,0].astype(np.int)]
    cmr = np.hstack((cmr, readnumsr[:,np.newaxis]))
    # The first column indexes into cm; we no longer need it.
    lof = lof[:,1:]
    lor = lor[:,1:]

    lo_fr = np.vstack((lof, lor))
    cm_fr = np.vstack((cmf, cmr))

    # return cm for major, cm for minor, concatenated
    # return one cm, one lo, major and minor concatenated, num_major

    if minor == 'N':
        minor = npr.choice([el for el in 'ACGT' if el != major])

    forward_major = np.zeros(4)
    forward_major['ACGT'.index(major)] = 1.0
    forward_major = np.tile(forward_major, (lof.shape[0], 1))
    forward_minor = np.zeros(4)
    forward_minor['ACGT'.index(minor)] = 1.0
    forward_minor = np.tile(forward_minor, (lof.shape[0], 1))

    reverse_major = np.zeros(4)
    reverse_major['TGCA'.index(major)] = 1.0
    reverse_major = np.tile(reverse_major, (lor.shape[0], 1))
    reverse_minor = np.zeros(4)
    reverse_minor['TGCA'.index(minor)] = 1.0
    reverse_minor = np.tile(reverse_minor, (lor.shape[0], 1))

    major_fr = np.vstack((forward_major, reverse_major))
    minor_fr = np.vstack((forward_minor, reverse_minor))

    cm_major = np.hstack((cm_fr, major_fr))
    cm_minor = np.hstack((cm_fr, minor_fr))
    all_cm = np.array([cm_major, cm_minor])
    # Put the 'read' dimension first, to enable support for masking.
    all_cm = np.swapaxes(all_cm, 0, 1).copy()
    all_lo = lo_fr
    gc.collect()
    return all_cm, all_lo


def main():
    print('#' + ' '.join(sys.argv))

    parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input', help='input HDF5 file')
    parser.add_argument('--num-data-threads', type=int,
                        help='number of threads to use for data processing',
                        default=2)
    parser.add_argument('--load-model', help='tensorflow model to load')
    args = parser.parse_args()

    dat = h5py.File(args.input, 'r')
    h5cm = dat['covariate_matrices']
    h5lo = dat['locus_observations']
    print('# loading all_majorminor')
    all_majorminor = get_major_minor(dat)
    print('# obtaining column names')
    colnames_str = dat.attrs['covariate_column_names']
    colnames = colnames_str.split(',')


    print('# getting covariate matrix keys')
    h5cm_keys = []
    for chrom, chrom_cm in h5cm.iteritems():
        for bam, bam_cm in chrom_cm.iteritems():
            print('# getting covariate matrix keys: {}'.format(bam))
            for locus, locus_cm in bam_cm.iteritems():
                spname = locus_cm.name.split('/')
                name = unicode('/'.join(spname[2:]))
                h5cm_keys.append(name)

    print('# getting locus observation keys')
    h5lo_keys = []
    for chrom, chrom_lo in h5lo.iteritems():
        for bam, bam_lo in chrom_lo.iteritems():
            print('# getting locus observation keys: {}'.format(bam))
            for locus, locus_lo in bam_lo.iteritems():
                spname = locus_lo.name.split('/')
                name = unicode('/'.join(spname[2:]))
                h5lo_keys.append(name)
    assert set(h5lo_keys) == set(h5cm_keys), "covariate matrix and locus observation keys differ"

    locus_keys = h5cm_keys
    arglist = get_args(h5cm_keys, all_majorminor)  # each element is (key, major, minor)
    num_args = len(arglist)
    dat.close()
    del h5cm, h5lo

    def produce_data(out_queue, in_queue, h5fn):
        dat = h5py.File(args.input, 'r')
        h5cm = dat['covariate_matrices']
        h5lo = dat['locus_observations']
        while True:
            locus, major, minor = in_queue.get()
            if minor == 'N':
                # If there is no minor allele (either because all bases were
                # called the same, or because two bases had the same number of
                # variant calls), choose one at random.
                other_bases = [base for base in 'ACGT' if base != major]
                minor = npr.choice(other_bases)
            cm, lo = get_cm_and_lo(locus, major, minor, h5cm, h5lo)
            cm = cm.astype(np.float32)
            lo = lo.astype(np.float32)
            out_queue.put(((cm, lo), np.ones(cm.shape[0])))
            del cm, lo
            gc.collect()

    num_data_processing_threads = args.num_data_threads
    data_queue = mp.Queue(256)
    input_queues = [mp.Queue(0) for i in range(num_data_processing_threads)]

    data_processes = [
        mp.Process(target=produce_data, args=(data_queue, input_queues[tid], args.input))
                       for tid in range(num_data_processing_threads)]
    for p in data_processes:
        p.start()

    def data_generator():
        args = np.array(arglist[:])
        while True:
            npr.shuffle(args)
            split_args = np.array_split(args, num_data_processing_threads)
            for tid, tid_args in enumerate(split_args):
                for tid_arg in tid_args:
                    input_queues[tid].put(tid_arg)

            while True:
                try:
                    ((cm, lo), ones) = data_queue.get()
                    yield ((cm, lo), ones)
                    del cm, lo
                    gc.collect()
                except Empty:
                    break
            

    ((cm, lo), _) = data_generator().next()
    num_cm_columns = cm.shape[2]

    cm_input = layers.Input(shape=(None, 2, num_cm_columns))
    masked_cm_input = layers.Masking(mask_value=MASK_VALUE)(cm_input)
    layer1 = layers.Dense(32, activation='softplus')(masked_cm_input)
    layer2 = layers.Dense(16, activation='softplus')(layer1)
    output_softmax = layers.Dense(4, activation='softmax')(layer2)
    nn_output = layers.Lambda(lambda x: tf.math.log(x))(output_softmax)
    num_f = 256

    nn_logprobs = tf.keras.Model(inputs=cm_input, outputs=nn_output)
    logpf = Likelihood(num_f)

    lo_input = layers.Input(shape=(None, 4))
    masked_lo_input = layers.Masking(mask_value=MASK_VALUE)(lo_input)

    likelihood = logpf([nn_output, masked_lo_input])

    ll_model = tf.keras.Model(inputs=[cm_input, lo_input], outputs=likelihood)
    ll_loss = LikelihoodLoss()
    ll_model.compile(optimizer='Adam', loss=ll_loss)
    if args.load_model:
        ll_model.load_weights(args.load_model)

    batch_size = 32
    output_types = ((tf.float32, tf.float32), tf.float32)
    data = tf.data.Dataset.from_generator(
        data_generator,
        output_types=output_types)
    data = data.padded_batch(
        batch_size=batch_size,
        padded_shapes=(((-1, 2, num_cm_columns), (-1, 4)), (-1,)),
        padding_values=((MASK_VALUE, MASK_VALUE), MASK_VALUE))
    data = data.prefetch(2*batch_size)

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        'error.model',load_weights_on_restart=True)
    num_epochs = 10000
    batches_per_epoch = 10
    ll_model.fit(data, epochs=num_epochs, steps_per_epoch=batches_per_epoch,
              callbacks=[checkpoint_callback])

if __name__ == '__main__':
    main()
