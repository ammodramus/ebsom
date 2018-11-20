from __future__ import division
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy.random as npr
import re
import tensorflow as tf
from scipy.special import logsumexp

import beta_with_spikes_integrated as bws


def complement(bases):
    return ''.join(['TGCA'['ACGT'.index(base)] for base in bases])


def get_base_covs(major, minor):
    major_comp = complement(major)
    minor_comp = complement(minor)
    maj_f_base_cov = np.zeros(4, dtype=np.float32)
    min_f_base_cov = np.zeros(4, dtype=np.float32)
    maj_r_base_cov = np.zeros(4, dtype=np.float32)
    min_r_base_cov = np.zeros(4, dtype=np.float32)
    maj_f_base_cov['ACGT'.index(major)] = 1
    min_f_base_cov['ACGT'.index(minor)] = 1
    maj_r_base_cov['ACGT'.index(major_comp)] = 1
    min_r_base_cov['ACGT'.index(minor_comp)] = 1
    return maj_f_base_cov, min_f_base_cov, maj_r_base_cov, min_r_base_cov


def make_neural_net(n_cols, n_cols_const, hidden_layer_sizes):
    if len(hidden_layer_sizes) < 1:
        raise ValueError('hidden_layer_sizes must contain at least one integer')

    n_bases = 4

    forward_inputs = tf.placeholder(tf.float32, (None, n_cols), name = 'for_inputs')
    reverse_inputs = tf.placeholder(tf.float32, (None, n_cols), name = 'rev_inputs')
    
    # note:  const input will have both the constant covariates and the "true" base
    major_forward_inputs_const = tf.placeholder(tf.float32, (1,n_cols_const+n_bases), name = 'maj_for_inputs_const')
    minor_forward_inputs_const = tf.placeholder(tf.float32, (1,n_cols_const+n_bases), name = 'min_for_inputs_const')
    major_reverse_inputs_const = tf.placeholder(tf.float32, (1,n_cols_const+n_bases), name = 'maj_rev_inputs_const')
    minor_reverse_inputs_const = tf.placeholder(tf.float32, (1,n_cols_const+n_bases), name = 'min_rev_inputs_const')

    # First, the inputs

    # Calculate number of parameters (i.e., num weights and biases)
    weight_shapes = [(n_cols, hidden_layer_sizes[0])]
    if len(hidden_layer_sizes) > 1:
        for i, hls in enumerate(hidden_layer_sizes):
            if i == 0:
                continue
            weight_shapes.append((hidden_layer_sizes[i-1], hls))
    weight_shapes.append((hidden_layer_sizes[-1], n_bases))
    bias_shapes = hidden_layer_sizes + [n_bases]
    
    const_weights_shape = (n_cols_const+n_bases, hidden_layer_sizes[0])
    
    total_num_params = (
        sum([shp[0]*shp[1] for shp in weight_shapes]) + sum(bias_shapes) +
        const_weights_shape[0]*const_weights_shape[1])

    # Create parameters tensor containing all of the weights and biases as a single placeholder array
    with tf.name_scope('params'):
        params = tf.placeholder(tf.float32, [total_num_params])

    # Define weights and biases in terms of indices to this parameters tensor.
    # This replaces the weight- and bias-creation below.
    weights = {}
    weights['hidden'] = []

    start = 0

    end = n_cols*hidden_layer_sizes[0] + start
    weights['hidden'].append(tf.reshape(params[start:end], [n_cols, hidden_layer_sizes[0]]))
    start = end
    
    end = start + hidden_layer_sizes[0]*(n_cols_const+n_bases)
    weights['const'] = tf.reshape(params[start:end], (n_cols_const+n_bases, hidden_layer_sizes[0]))
    start = end

    for i in range(1, len(hidden_layer_sizes)):
        end = hidden_layer_sizes[i-1]*hidden_layer_sizes[i] + start
        weights['hidden'].append(tf.reshape(params[start:end], (hidden_layer_sizes[i-1],hidden_layer_sizes[i])))
        start = end
    end = start + hidden_layer_sizes[-1]*n_bases
    weights['out'] = tf.reshape(params[start:end], (hidden_layer_sizes[-1], n_bases))
    start = end

    # Create the biases
    biases = {}
    biases['hidden'] = []
    for hls in hidden_layer_sizes:
        end = start + hls
        biases['hidden'].append(params[start:end])
        start = end
    end = start + n_bases
    biases['out'] = params[start:end]
    start = end
    
    assert start == total_num_params, 'start: {}, total_num_params: {}'.format(start, total_num_params)

    # Major
    hidden_layers = []
    # Assume there is at least one hidden layer...
    with tf.name_scope('firstlayermajor'):
        forward_changing_input0 = tf.matmul(forward_inputs,weights['hidden'][0], name='forward_changing_input0')
        major_forward_const_input0 = tf.matmul(major_forward_inputs_const,weights['const'], name='major_forward_const_input0')
        major_forward_input0 = forward_changing_input0 + major_forward_const_input0
        
        reverse_changing_input0 = tf.matmul(reverse_inputs,weights['hidden'][0], name='reverse_changing_input0')
        major_reverse_const_input0 = tf.matmul(major_reverse_inputs_const,weights['const'], name='major_reverse_const_input0')
        major_reverse_input0 = reverse_changing_input0 + major_reverse_const_input0
        
        input0 = tf.concat((major_forward_input0, major_reverse_input0), axis = 0) + biases['hidden'][0]
        first_layer = tf.nn.softplus(input0)
        
    hidden_layers.append(first_layer)
            
    # Calculate the remaining hidden layers
    for i in range(1, len(hidden_layer_sizes)):
        with tf.name_scope('layer_{}'.format(i+1)):
            prev_layer = hidden_layers[i-1]
            layer = tf.nn.softplus(tf.add(tf.matmul(prev_layer, weights['hidden'][i]), biases['hidden'][i]))
            hidden_layers.append(layer)
    with tf.name_scope('output'):
        out_layer = tf.matmul(hidden_layers[-1], weights['out']) + biases['out']
    with tf.name_scope('logprobs'):
        logprobs_major = tf.nn.log_softmax(out_layer)

    # Minor
    hidden_layers = []
    # Assume there is at least one hidden layer...
    with tf.name_scope('firstlayerminor'):
        forward_changing_input0 = tf.matmul(forward_inputs,weights['hidden'][0], name='forward_changing_input0')
        minor_forward_const_input0 = tf.matmul(minor_forward_inputs_const,weights['const'], name='minor_forward_const_input0')
        minor_forward_input0 = forward_changing_input0 + minor_forward_const_input0
        
        reverse_changing_input0 = tf.matmul(reverse_inputs,weights['hidden'][0], name='reverse_changing_input0')
        minor_reverse_const_input0 = tf.matmul(minor_reverse_inputs_const,weights['const'], name='minor_reverse_const_input0')
        minor_reverse_input0 = reverse_changing_input0 + minor_reverse_const_input0
        
        input0 = tf.concat((minor_forward_input0, minor_reverse_input0), axis = 0) + biases['hidden'][0]
        first_layer = tf.nn.softplus(input0)
    hidden_layers.append(first_layer)
    # Calculate the remaining hidden layers
    for i in range(1, len(hidden_layer_sizes)):
        prev_layer = hidden_layers[i-1]
        layer = tf.nn.softplus(tf.add(tf.matmul(prev_layer, weights['hidden'][i]), biases['hidden'][i]))
        hidden_layers.append(layer)
    out_layer = tf.matmul(hidden_layers[-1], weights['out']) + biases['out']
    logprobs_minor = tf.nn.log_softmax(out_layer)
    
    return (params, forward_inputs, reverse_inputs, major_forward_inputs_const, minor_forward_inputs_const,
            major_reverse_inputs_const, minor_reverse_inputs_const, logprobs_major, logprobs_minor)


def get_ll_gradient_and_inputs(n_cols, n_cols_const, hidden_layer_sizes, nfreqs):
    ll_aux = make_neural_net(n_cols, n_cols_const, hidden_layer_sizes)
    (nn_params, forward_inputs, reverse_inputs, major_forward_inputs_const, minor_forward_inputs_const,
            major_reverse_inputs_const, minor_reverse_inputs_const, logprobs_major, logprobs_minor) = ll_aux
    logf = tf.placeholder(tf.float32, [nfreqs], name = 'logf')
    log1mf = tf.placeholder(tf.float32, [nfreqs], name = 'log1mf')
    lpf = tf.placeholder(tf.float32, [nfreqs], name = 'lpf')
    counts = tf.placeholder(tf.float32, [None, 4], name = 'counts')
    a_term = tf.add(
            tf.expand_dims(tf.expand_dims(logf, axis = -1), axis = -1),
            tf.expand_dims(logprobs_minor, axis = 0)
            )
    b_term = tf.add(
            tf.expand_dims(tf.expand_dims(log1mf, axis = -1), axis = -1),
            tf.expand_dims(logprobs_major, axis = 0)
            )
    lse = tf.reduce_logsumexp(tf.stack((a_term, b_term)), axis = 0)
    tmp = tf.multiply(lse, counts)
    b_sums = tf.reduce_sum(tmp, axis = [1,2], name = 'freq_logposteriors')
    f_posts = lpf + b_sums
    ll = tf.reduce_logsumexp(f_posts, name = 'loglike')
    nn_grads = tf.gradients(ll, nn_params, name = 'nn_grads')
    return (ll, nn_grads, nn_params, forward_inputs, reverse_inputs, major_forward_inputs_const,
            minor_forward_inputs_const, major_reverse_inputs_const,  minor_reverse_inputs_const,
            counts, logf, log1mf, lpf, b_sums)



def loglike_and_gradient_wrapper(params, forward_cov, reverse_cov,
        forward_const_cov, reverse_const_cov, counts_forward, counts_reverse,
        major, minor, num_pf_params, logf, log1mf, freqs, windows, ll_aux,
        sess):
    '''
    params are current parameter values, first num_pf_params are for the AFD
    forward_cov is the covariates for the reads in forward orientation
    reverse_cov is the covariates for the reads in reverse orientation
    forward_const_cov is the covariates for all of the reads in the forward orientation
    reverse_const_cov is the covariates for all of the reads in the reverse orientation
    major, minor are major and minor bases
    num_pf_params, freqs, windows for lpf
    ll_aux is all the variables that return from get_ll_tf
    sess is a tf.Session
    '''
    (ll_tf, nn_grads_tf, nn_params_tf, forward_inputs_tf, reverse_inputs_tf, major_forward_inputs_const_tf,
            minor_forward_inputs_const_tf, major_reverse_inputs_const_tf,  minor_reverse_inputs_const_tf,
            counts_tf, logf_tf, log1mf_tf, logpf_tf, b_sums_tf) = ll_aux
    pf_pars = params[:num_pf_params]
    nn_pars = params[num_pf_params:]

    # Get the AFD log-probability distribution
    lpf_np = bws.get_lpf(pf_pars, freqs, windows)

    if minor == 'N':

        # Have to average the likelihood over all the potential minor alleles
        alt_minors = [base for base in 'ACGT' if base != major]

        gradients = []
        lls = []
        for alt_minor in alt_minors:
            tll, tgrad = loglike_and_gradient_wrapper(params, forward_cov,
                    reverse_cov, forward_const_cov, reverse_const_cov,
                    counts_forward, counts_reverse, major, alt_minor,
                    num_pf_params, logf, log1mf, freqs, windows, ll_aux, sess)
            lls.append(tll)
            gradients.append(tgrad)
        lls = np.array(lls)
        gradients = np.array(gradients)
        sign_grads = np.sign(gradients)
        logabs_grads = np.log(np.abs(gradients))
        logdenom = logsumexp(lls)
        pos_log_grads = (logabs_grads+lls[:,np.newaxis])
        neg_log_grads = (logabs_grads+lls[:,np.newaxis])
        pos_log_grads[sign_grads < 0] = -np.inf
        neg_log_grads[sign_grads >= 0] = -np.inf
        grads = np.exp(logsumexp(pos_log_grads, axis = 0)-logdenom) - np.exp(logsumexp(neg_log_grads, axis = 0)-logdenom)
        ll = logdenom - np.log(3.0)   # divided by 3 for average
        return ll, grads

    else:
        major_forward_base_cov, minor_forward_base_cov, major_reverse_base_cov, minor_reverse_base_cov = (
            get_base_covs(major, minor))

        major_forward_const = np.concatenate((major_forward_base_cov, forward_const_cov))[np.newaxis,:]
        minor_forward_const = np.concatenate((minor_forward_base_cov, forward_const_cov))[np.newaxis,:]
        major_reverse_const = np.concatenate((major_reverse_base_cov, reverse_const_cov))[np.newaxis,:]
        minor_reverse_const = np.concatenate((minor_reverse_base_cov, reverse_const_cov))[np.newaxis,:]

        combined_counts = np.concatenate((counts_forward, counts_reverse), axis = 0)

        feed_dict = {
                nn_params_tf: nn_pars,
                forward_inputs_tf: forward_cov,
                reverse_inputs_tf: reverse_cov,
                major_forward_inputs_const_tf: major_forward_const,
                minor_forward_inputs_const_tf: minor_forward_const,
                major_reverse_inputs_const_tf: major_reverse_const,
                minor_reverse_inputs_const_tf: minor_reverse_const,
                counts_tf: combined_counts,
                logf_tf: logf,
                log1mf_tf: log1mf,
                logpf_tf: lpf_np
                }
        ll, nngrads, b_sums = sess.run([ll_tf, nn_grads_tf, b_sums_tf], feed_dict=feed_dict)
        nngrads = nngrads[0]
        grads = np.zeros_like(params)
        grads[num_pf_params:] = nngrads

        # Calculate gradient for pf_params
        eps = 1e-7
        pf = np.exp(lpf_np)
        for i in range(num_pf_params):
            pf_pars[i] += eps  # inc by eps
            pf2 = np.exp(bws.get_lpf(pf_pars,freqs,windows))
            pf_pars[i] -= eps  # fix the eps
            dpfs = (pf2-pf)/eps
            filt = dpfs >= 0
            if np.any(filt):
                # Probably not any faster to use numexpr here
                # pos_log = logsumexp(np.log(dpfs[filt]) + b_sums[filt]) - ll
                pos_log = logsumexp(np.log(np.abs(dpfs[filt])) + b_sums[filt]) - ll
            else:
                pos_log = -np.inf
            if np.any(~filt):
                neg_log = logsumexp(np.log(np.abs(dpfs[~filt])) + b_sums[~filt]) - ll
            else:
                neg_log = -np.inf
            grads[i] = np.exp(pos_log) - np.exp(neg_log)

        return ll, grads



if __name__ == '__main__':
    x = get_ll_gradient_and_inputs(8, 17, [20,20],200)
    import pdb; pdb.set_trace()
