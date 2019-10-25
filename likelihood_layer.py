from __future__ import print_function
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers


def _get_psmc_times(n, tmax, conc_factor = 10):
    t = 1.0/conc_factor*(np.exp(
        np.arange(1,n+1, dtype = np.float)/n * np.log(1 + conc_factor*tmax))-1)
    return t

# The TF betainc gradient isn't implemented, so we use symmetric finite
# differences to estimate it.
@tf.custom_gradient
def betainc(A, B, v):
    # Casting to 64-bit is necessary for accuarcy
    A = tf.cast(A, tf.float64)
    B = tf.cast(B, tf.float64)
    v = tf.cast(v, tf.float64)
    ret = tf.math.betainc(A, B, v)
    epsilon = 1e-8
    def grad_fn(dy):
        a_inc = tf.math.betainc(A + epsilon, B, v)
        a_dec = tf.math.betainc(A - epsilon, B, v)
        a_grad = (a_inc-a_dec)/(2.0*epsilon)

        b_inc = tf.math.betainc(A, B + epsilon, v)
        b_dec = tf.math.betainc(A, B - epsilon, v)
        b_grad = (b_inc-b_dec)/(2.0*epsilon)

        v_grad = None   # Not implemented
        dy = tf.cast(dy, tf.float64)
        return [tf.cast(tf.tensordot(dy,a_grad,axes=[0,0]), tf.float32),
                tf.cast(tf.tensordot(dy,b_grad,axes=[0,0]), tf.float32), v_grad]
    return tf.cast(ret, tf.float32), grad_fn


class Likelihood(layers.Layer):
    '''
    Implements a tf.keras Layer that combines the allele frequency spectrum
    with the log-likelihood outputs from the sequencing-error neural network.
    '''

    def __init__(self, num_f, conc_factor=10, **kwargs):
        super(Likelihood, self).__init__(**kwargs)
        self.num_f = num_f
        self.conc_factor = conc_factor
        self.supports_masking = True

    def build(self, inp):
        self.params = self.add_weight('lpf_params', shape=(3,),
                                      initializer='random_normal',
                                      trainable=True)
        self.set_window_boundaries()
        self.set_freqs()
        super(Likelihood, self).build(inp)
        self.built = True
        pass

    def call(self, inp):
        v = self.window_boundaries
        # lA is log A, lB is log B, and expitz is expit(z)
        params = self.params
        lA = params[0]
        lB = params[1]
        expitz = params[2]

        A = tf.math.exp(lA)  # translate lA from (-inf, inf) to (0, inf)
        B = tf.math.exp(lB)  # translate lB from (-inf, inf) to (0, inf)
        z = tf.math.sigmoid(expitz)  # translate expitz from (-inf, inf) to (0,1)


        If_l = betainc(A, B, v)
        If_h = betainc(A,B, 1-v)
        diff_Ifl = If_l[1:]-If_l[:-1]
        diff_If_h = If_l[1:]-If_l[:-1]
        diff_If_h_rev = (If_h[::-1][1:] - If_h[::-1][:-1])[::-1]
        pf = (diff_If_h + diff_If_h_rev)*(1-z)
        lpf = tf.concat([tf.expand_dims(tf.math.log(z), axis=0), tf.math.log(pf)], axis=0, name='concat')

        output, masked_lo_input = inp
        output_major, output_minor = tf.split(output, 2, axis=2)

        logf_plus_minor_lls = (
            tf.reshape(self.logf, (1, 1, self.num_f, 1)) + output_minor)
        logf_plus_major_lls = (
            tf.reshape(self.log1mf, (1, 1, self.num_f, 1)) + output_major)
        logaddexp_terms = tf.math.reduce_logsumexp(
            tf.concat(
                [tf.expand_dims(logf_plus_minor_lls, axis=-1),
                 tf.expand_dims(logf_plus_major_lls, axis=-1)], axis=-1), axis=-1)
        logaddexp_terms_mult_lo = tf.multiply(logaddexp_terms,
                                              tf.expand_dims(masked_lo_input, axis=2))
        # Axis 1 corresponds to the reads; axis -1 corresponds to the bases.
        f_ll = tf.math.reduce_sum(logaddexp_terms_mult_lo, axis=[1,-1])
        posterior_logprobs = tf.expand_dims(lpf, axis=0) + f_ll
        return posterior_logprobs

    def set_window_boundaries(self):
        f = np.zeros(self.num_f)
        f[0] = 0.0
        f[1:] = _get_psmc_times(self.num_f-1,0.5, self.conc_factor)
        f = tf.constant(f, dtype='float32')
        self.window_boundaries = f

    def set_freqs(self):
        v = self.window_boundaries
        freq0 = tf.constant((0.,))
        f = tf.concat((freq0, (v[:-1]+v[1:])/2.0), 0)
        self.freqs = tf.Variable(initial_value=f, trainable=False,
                                 name='freqs')
        self.logf = tf.math.log(f)
        self.log1mf = tf.math.log(1.0-f)

@tf.function
def likelihood_loss(log_posts):
    return -1.0*tf.keras.backend.mean(tf.math.reduce_logsumexp(log_posts,
                                                               axis=1))
