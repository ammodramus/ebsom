import h5py
import numpy as np
import numpy.random as npr
import beta_with_spikes_integrated as bws
import sys
import argparse
import datetime

import h5py_util

import tensorflow as tf
import tensorflow.keras.layers as layers

tf.enable_eager_execution()


def get_window_boundaries(num_f, conc_factor = 10):
    f = np.zeros(num_f)
    f[0] = 0.0
    f[1:] = get_psmc_times(num_f-1,0.5, conc_factor)
    f = tf.constant(f, dtype='float32')
    return f

def get_psmc_times(n, tmax, conc_factor = 10):
    t = 1.0/conc_factor*(np.exp(
        np.arange(1,n+1, dtype = np.float)/n * np.log(1 + conc_factor*tmax))-1)
    return t

def get_freqs(num_f, conc_factor = 10):
    v = get_window_boundaries(num_f, conc_factor)
    f = np.concatenate(((0,), (v[:-1]+v[1:])/2.0))
    f = tf.constant(f, dtype='float32')
    return f


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


def get_lpf(params, window_boundaries):
    '''
    calculate log-probabilities of the minor allele frequency under a
    beta-with-spike model

    params is a 3-vector, lA, lB, and expitz, where lA and lB are logs of the
    beta parameters, and logit(expitz) is the probability that a locus is
    non-polymorphic.

    the window boundaries run from 0 to 0.5; probabilities are integrated
    within each window running from 0 to 0.5, and also from 0.5 to 1.0 (folded
    over 0.5), and then summed. the distribution runs from 0 to 0.5 because
    this is a minor allele frequency distribution.
    '''
    v = window_boundaries
    # lA is log A, lB is log B, and expitz is expit(z)
    lA = params[0]
    lB = params[1]
    expitz = params[2]

    A = tf.math.exp(lA)  # translate lA from (-inf, inf) to (0, inf)
    B = tf.math.exp(lB)  # translate lB from (-inf, inf) to (0, inf)
    z = tf.math.sigmoid(expitz)  # translate expitz from (-inf, inf) to (0,1)


    If_l = tf.math.betainc(A, B, v)
    If_h = tf.math.betainc(A,B, 1-v)
    diff_Ifl = If_l[1:]-If_l[:-1]
    diff_If_h = If_l[1:]-If_l[:-1]
    diff_If_h_rev = (If_h[::-1][1:] - If_h[::-1][:-1])[::-1]
    pf = (diff_If_h + diff_If_h_rev)*(1-z)
    lpf = tf.concat([tf.expand_dims(tf.math.log(z), axis=0), tf.math.log(pf)], axis=0, name='concat')
    return lpf

print '#' + ' '.join(sys.argv)

parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('input', help = 'input HDF5 file')
args = parser.parse_args()

dat = h5py.File(args.input, 'r')
print '# loading all_majorminor'
all_majorminor = get_major_minor(dat)
print '# obtaining column names'
colnames_str = dat.attrs['covariate_column_names']
colnames = colnames_str.split(',')

h5cm = dat['covariate_matrices']
h5lo = dat['locus_observations']

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
arglist = get_args(h5cm_keys, all_majorminor)  # each element is (key, major, minor)
num_args = len(arglist)

def get_cm_and_lo(key, major, minor):
    cm = h5cm[key][:]
    lo = h5lo[key]
    cur_idx = 0
    # forward first
    lof1 = lo['f1'][:]
    lof2 = lo['f2'][:]
    lor1 = lo['r1'][:]
    lor2 = lo['r2'][:]

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

    forward_major = np.zeros(5)
    forward_major['ACGTN'.index(major)] = 1.0
    forward_major = np.tile(forward_major, (lof.shape[0], 1))
    forward_minor = np.zeros(5)
    forward_minor['ACGTN'.index(minor)] = 1.0
    forward_minor = np.tile(forward_minor, (lof.shape[0], 1))

    reverse_major = np.zeros(5)
    reverse_major['TGCAN'.index(major)] = 1.0
    reverse_major = np.tile(reverse_major, (lor.shape[0], 1))
    reverse_minor = np.zeros(5)
    reverse_minor['TGCAN'.index(minor)] = 1.0
    reverse_minor = np.tile(reverse_minor, (lor.shape[0], 1))

    return (
        cmf, cmr, lof, lor, forward_major, forward_minor, reverse_major,
        reverse_minor
    )


def get_cm_and_lo_alltogethernow(key, major, minor):
    cm = h5cm[key][:]
    lo = h5lo[key]
    cur_idx = 0
    # forward first
    lof1 = lo['f1'][:]
    lof2 = lo['f2'][:]
    lor1 = lo['r1'][:]
    lor2 = lo['r2'][:]

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

    all_lo = lo_fr

    return all_cm, all_lo


class LogPF(layers.Layer):

    def __init__(self, num_f):
        super(LogPF, self).__init__()
        self.num_f = num_f
        self.freqs = get_freqs(self.num_f)

    def build(self):
        self.par = self.add_weight(shape=(3,), initializer='random_normal',
                                   trainable=True)

    def call(self):
        # TODO here...




def data_generator():
    while True:
        yield (tuple([el.astype(np.float32) for el in
                     get_cm_and_lo(*arglist[9])]), 1.0)


cm, lo = get_cm_and_lo_alltogethernow(*arglist[9])

cm_input = layers.Input(shape=(2, None, cm.shape[2]))
layer1 = layers.Dense(32, activation='softplus')(cm_input)
layer2 = layers.Dense(32, activation='softplus')(layer1)
output_softmax = layers.Dense(4, activation='softmax')(layer2)
output = layers.Lambda(lambda x: tf.math.log(x))(output_softmax)
output_major, output_minor = tf.split(output, 2, axis=1)
#output_major = tf.squeeze(output_major, axis=1)
#output_minor = tf.squeeze(output_minor, axis=1)

num_f = 256
freqs = get_freqs(num_f)
logf = tf.math.log(freqs)
log1mf = tf.math.log(1.0-freqs)
window_boundaries = get_window_boundaries(num_f)

distn_pars = tf.keras.backend.variable(value=np.zeros(3),
                                       name='distn_variables',
                                       dtype='float32')
lpf = get_lpf(distn_pars, window_boundaries)


nn_logprobs = tf.keras.Model(inputs=cm_input, outputs=[output_major, output_minor])

logf_plus_minor_lls = tf.reshape(logf, (1, num_f, 1, 1)) + output_minor
logf_plus_major_lls = tf.reshape(log1mf, (1, num_f, 1, 1)) + output_major
logaddexp_terms = tf.math.reduce_logsumexp(
    tf.concat(
        [tf.expand_dims(logf_plus_minor_lls, axis=-1),
         tf.expand_dims(logf_plus_major_lls, axis=-1)], axis=-1), axis=-1)
f_ll = tf.math.reduce_sum(logaddexp_terms, axis=[-2,-1])
tmp = tf.expand_dims(lpf, axis=0) + f_ll
final_ll = tf.math.reduce_logsumexp(tmp, axis=1)

ll_model = tf.keras.Model(inputs=cm_input, outputs=final_ll)

import pdb; pdb.set_trace()




import pdb; pdb.set_trace()


########################
# The model
########################


'''
cmf_input = layers.Input(shape=(None, cmf.shape[1]), name='cmf_input')

forward_major_input = layers.Input(shape=(None, forward_major.shape[1]))
cmf_input_major = layers.Concatenate(axis=-1, name='cmf_input_concatenated')([cmf_input, forward_major_input])
forward_minor_input = layers.Input(shape=(None, forward_minor.shape[1]))
cmf_input_minor = layers.Concatenate(axis=-1)([cmf_input, forward_minor_input])

cmr_input = layers.Input(shape=(None, cmr.shape[1]))

reverse_major_input = layers.Input(shape=(None, reverse_major.shape[1]))
cmr_input_major = layers.Concatenate(axis=-1)([cmr_input, reverse_major_input])
reverse_minor_input = layers.Input(shape=(None, reverse_minor.shape[1]))
cmr_input_minor = layers.Concatenate(axis=-1)([cmr_input, reverse_minor_input])

layer1 = layers.Dense(32, activation='softplus')([cmf_input_major,
                                                  cmf_input_minor,
                                                  cmr_input_major,
                                                  cmr_input_minor])
layer2 = layers.Dense(32, activation='softplus')(layer1)
output_layer = layers.Dense(4, activation='softmax')(layer2)
output_layer = layers.layers.Lambda(lambda x: tf.math.log(x))(output_layer)

layer1_cmf_major = layer1(cmf_input_major)
layer1_cmf_minor = layer1(cmf_input_minor)
layer1_cmr_major = layer1(cmr_input_major)
layer1_cmr_minor = layer1(cmr_input_minor)



lof_input = layers.Input(shape=(None, lof.shape[1]))
lof_lls_major = layers.Multiply()([lof_input, cmf_major_output])
lof_lls_minor = layers.Multiply()([lof_input, cmf_minor_output])

lor_input = layers.Input(shape=(None, lor.shape[1]))
lor_lls_major = layers.Multiply()([lor_input, cmr_major_output])
lor_lls_minor = layers.Multiply()([lor_input, cmr_minor_output])

num_f = 256
freqs = get_freqs(num_f)
logf = tf.math.log(freqs)
log1mf = tf.math.log(1.0-freqs)
window_boundaries = get_window_boundaries(num_f)

distn_pars = tf.keras.backend.variable(value=np.zeros(3),
                                       name='distn_variables',
                                       dtype='float32')

lpf = get_lpf(distn_pars, window_boundaries)

lo_lls_minor = tf.concat([lof_lls_minor, lor_lls_minor], axis=1)
lo_lls_minor = tf.expand_dims(lo_lls_minor, axis=1)
lo_lls_major = tf.concat([lof_lls_major, lor_lls_major], axis=1)
lo_lls_major = tf.expand_dims(lo_lls_major, axis=1)

logf_plus_lo_lls_minor = tf.reshape(logf, [1,num_f,1,1]) + lo_lls_minor
logf_plus_lo_lls_minor = tf.expand_dims(logf_plus_lo_lls_minor, axis=-1)
log1mf_plus_lo_lls_major = tf.reshape(log1mf, [1,num_f,1,1]) + lo_lls_major
log1mf_plus_lo_lls_major = tf.expand_dims(log1mf_plus_lo_lls_major, axis=-1)

tmp = tf.concat([logf_plus_lo_lls_minor, log1mf_plus_lo_lls_major],
              axis=-1)
logaddexp_logf_log1mf_terms = tf.math.reduce_logsumexp(tmp, axis=-1)
f_terms = tf.reduce_sum(logaddexp_logf_log1mf_terms, axis=[-2,-1])

tmp = f_terms + tf.expand_dims(lpf, axis=0)
final_ll = tf.math.reduce_logsumexp(tmp, axis=1, name='final_ll')

all_inputs = [cmf_input, cmr_input, lof_input, lor_input, forward_major_input,
              forward_minor_input, reverse_major_input, reverse_minor_input]

ll_loss = lambda y_t, y_p: -y_p

model = tf.keras.Model(inputs=all_inputs, outputs=final_ll)
model.compile(optimizer='adam',
              loss=ll_loss)

output_types = tuple([tf.float32 for _ in range(len(all_inputs))])
data = tf.data.Dataset.from_generator(
    data_generator,
    output_types=output_types)
data = data.batch(20)
model.fit(data)
'''
