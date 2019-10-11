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
import scipy

import beta_with_spikes_integrated as bws
from likelihood_layer import Likelihood, LikelihoodLoss
from run_model import make_model, get_cm_and_lo

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
    cm_input, lo_input, log_posteriors = make_model(num_covariates,
                                                args.num_frequencies)
    ll_model = tf.keras.Model(inputs=[cm_input, lo_input],
                              outputs=log_posteriors)
    ll_model.load_weights(args.weights)

    freqs = ll_model.get_layer('log_posteriors').get_weights()[1]
    freqs_str = '#' + '\t'.join(map(lambda x: '{:.8e}'.format(x), freqs))
    print(freqs_str)

    header = '\t'.join(['best_freq', 'bam', 'chromosome', 'locus']
                       + ['freq'+str(i) for i in range(len(freqs))])
    for loc_key, major, minor in arglist:
        chrom, bam, locus = loc_key.split('/')
        cm, lo = get_cm_and_lo(loc_key, major, minor, h5cm, h5lo)
        # We run the model one locus at a time.
        log_posteriors = ll_model([cm[np.newaxis,:],
                                   lo[np.newaxis,:]]).numpy()[0]
        total_ll = scipy.special.logsumexp(log_posteriors)
        log_posteriors_norm = log_posteriors - total_ll
        best_freq = freqs[np.argmax(log_posteriors_norm)]
        output_line = '\t'.join([bam, chrom, locus] + 
            map(lambda x: '{:.8e}'.format(x), [best_freq] +
                                    list(log_posteriors_norm)))
        print(output_line)


if __name__ == '__main__':
    main()
