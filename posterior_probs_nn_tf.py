import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import scipy.optimize as opt
import h5py
import numpy as np
import numpy.random as npr
import afd
import beta_with_spikes_integrated as bws
import util
import sys
import argparse
import datetime

import h5py_util
import neural_net as nn

import tensorflow_neural_net as tfnn
import tensorflow as tf

from scipy.special import logsumexp

import argparse

parser = argparse.ArgumentParser(
        description='Get posterior probabilities from allele-frequency distribution and neural parameters',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('input', help = 'input HDF5 data file')
parser.add_argument('params', help = 'file with parameters, one per line, allele-frequency distribution parameters first')
parser.add_argument('hiddenlayersizes', type = int, nargs = '+', help = 'sizes of hidden layers')
args = parser.parse_args()


num_pf_params = 3
num_f = 100



pars = np.loadtxt(args.params)
pf_pars = pars[:num_pf_params]
nn_pars = pars[num_pf_params:]

freqs = bws.get_freqs(num_f)
windows = bws.get_window_boundaries(num_f)
lf = np.log(freqs)
l1mf = np.log(1-freqs)
lpf = bws.get_lpf(pf_pars, freqs, windows)

dat = h5py.File(args.input, 'r')
print '# loading all_majorminor'
all_majorminor = h5py_util.get_major_minor(dat)
print '# obtaining column names'
colnames_str = dat.attrs['covariate_column_names']
colnames = colnames_str.split(',')

h5cm = dat['covariate_matrices']
h5lo = dat['locus_observations']
h5mm = dat['major_minor']

h5cm_keys = []
for chrom, chrom_cm in h5cm.iteritems():
    for bam, bam_cm in chrom_cm.iteritems():
        for locus, locus_cm in bam_cm.iteritems():
            spname = locus_cm.name.split('/')
            cm_key = unicode('/'.join(spname[2:]))
            break
        break
    break

cm = h5cm[cm_key][:]
lo = h5lo[cm_key]
lo = [[lo['f1'][:], lo['f2'][:]], [lo['r1'][:], lo['r2'][:]]]
num_inputs = nn.get_major_minor_cm_and_los(cm,lo, 'C', 'T')[0].shape[1]

logprobs_aux = tfnn.get_ll_and_grads_tf(num_inputs, args.hiddenlayersizes, num_f, return_f_posteriors = True)
params, major_inputs, minor_inputs, counts, logf, log1mf, logpf, ll, grads, b_sums, keep_prob_tf, f_posts = logprobs_aux
total_num_params = int(params.shape[0])
init = tf.global_variables_initializer()
session = tf.Session()
session.run(init)

if total_num_params+num_pf_params != pars.shape[0]:
    raise ValueError('incorrect number of parameters: expected {}, got {}'.format(
        total_num_params+num_pf_params, pars.shape[0]))


for chrom in h5lo.keys():
    for bam in h5lo[chrom].keys():
        loci = sorted(h5lo[chrom][bam].keys(), key = lambda x: int(x))
        bam_mm = h5mm[chrom][bam][:]
        for locus in loci:
            cm = h5cm[chrom][bam][locus][:]
            lo = h5lo[chrom][bam][locus]
            lo = [[lo['f1'][:], lo['f2'][:]], [lo['r1'][:], lo['r2'][:]]]
            # For each locus, calculate logprobs for major and minor
            # Use those to get loglikelihood of different fs
            # Add to that the logprior, got logposterior
            major, minor = bam_mm[int(locus)]
            major, minor = str(major), str(minor)
            if minor != 'N':
                major_cm, minor_cm, all_los = nn.get_major_minor_cm_and_los(cm,lo, major, minor)
                feed_dict = {params: nn_pars, major_inputs:major_cm, minor_inputs:minor_cm,
                        logf:lf, log1mf:l1mf, logpf:lpf, counts:all_los[:,1:], keep_prob_tf:1.0}
                logpost_fs = session.run(f_posts, feed_dict = feed_dict)
            else:
                minor_cms = []
                for alt_minor in [base for base in 'ACGT' if base != major]:
                    # Note: major_cm and all_los will be the same every time
                    major_cm, minor_cm, all_los = nn.get_major_minor_cm_and_los(cm,lo, major, alt_minor)
                    minor_cms.append(minor_cm)
                lpostfs = []
                for mcm in minor_cms:
                    feed_dict = {params: nn_pars, major_inputs:major_cm, minor_inputs:mcm,
                            logf:lf, log1mf:l1mf, logpf:lpf, counts:all_los[:,1:], keep_prob_tf:1.0}
                    m_logpost_fs = session.run(f_posts, feed_dict = feed_dict)
                    lpostfs.append(m_logpost_fs)
                logpost_fs = logsumexp(lpostfs, axis = 0)

            # Print results
            res = '{}\t{}\t{}\t{}'.format(chrom, bam, locus, '\t'.join([str(el) for el in logpost_fs]))
            print res
