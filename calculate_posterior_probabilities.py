from __future__ import print_function, division
import sys
import argparse
import datetime
import gc
import multiprocessing as mp
try:
    from Queue import Empty
except ImportError:
    # Python 3
    from asyncio import QueueEmpty as Empty

import h5py
import numpy as np
import numpy.random as npr
import tensorflow as tf
import tensorflow.keras.layers as layers

import beta_with_spikes_integrated as bws
from likelihood_layer import Likelihood, LikelihoodLoss

from run_model import make_model

class DebugCallback(tf.keras.callbacks.Callback):
  def on_train_batch_end(self, batch, logs=None):
    print('Training: batch {} ends at {}'.format(batch, datetime.datetime.now().time()))


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
    parser.add_argument('input', help='checkpoint directory of saved weights')
    parser.add_argument('weights', help='input model weights')
    parser.add_argument('--num-frequencies', type=int, default=128,
                        help='number of discrete frequencies')
    parser.add_argument('--num-data-threads', type=int,
                        help='number of threads to use for data processing',
                        default=2)
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

    cm, lo = get_cm_and_lo(arglist[0][0], arglist[0][1], arglist[0][2], h5cm,
                           h5lo)
    cm = cm.astype(np.float32)
    lo = lo.astype(np.float32)
    num_covariates = cm.shape[2]
    cm_input, lo_input, likelihood = make_model(num_covariates,
                                                args.num_frequencies)
    ll_model = tf.keras.Model(inputs=[cm_input, lo_input], outputs=likelihood)
    ll_model.load_weights(args.weights)

    log_posteriors = ll_model.get_layer('log_posteriors').output
    log_post_model = tf.keras.Model(inputs=[cm_input, lo_input],
                                    outputs=log_posteriors)
    import pdb; pdb.set_trace()


if __name__ == '__main__':
    main()
