from __future__ import print_function

import h5py
import numpy as np

import neural_net as nn

import tensorflow as tf

import numpy.random as npr

import time

import beta_with_spikes_integrated as bws



def make_neural_net(n_input, hidden_layer_sizes):
    if len(hidden_layer_sizes) < 1:
        raise ValueError('hidden_layer_sizes must contain at least one integer')

    n_bases = 4

    major_inputs = tf.placeholder(tf.float64, (None, n_input))
    minor_inputs = tf.placeholder(tf.float64, (None, n_input))

    # First, the inputs

    # Calculate number of parameters (i.e., num weights and biases)
    weight_shapes = [(n_input, hidden_layer_sizes[0])]
    if len(hidden_layer_sizes) > 1:
        for i, hls in enumerate(hidden_layer_sizes):
            if i == 0:
                continue
            weight_shapes.append((hidden_layer_sizes[i-1], hls))
    weight_shapes.append((hidden_layer_sizes[-1], n_bases))
    bias_shapes = hidden_layer_sizes + [n_bases]
    total_num_params = sum([shp[0]*shp[1] for shp in weight_shapes]) + sum(bias_shapes)

    # Create parameters tensor containing all of the weights and biases as a single placeholder array
    params = tf.placeholder(tf.float64, [total_num_params])

    # Define weights and biases in terms of indices to this parameters tensor.
    # This replaces the weight- and bias-creation below.
    weights = {}
    weights['hidden'] = []

    start = 0

    end = n_input*hidden_layer_sizes[0] + start
    weights['hidden'].append(tf.reshape(params[start:end], [n_input, hidden_layer_sizes[0]]))
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

    assert start == total_num_params

    # Major
    hidden_layers = []
    # Assume there is at least one hidden layer...
    #hidden_layers.append(tf.nn.softplus(tf.add(tf.matmul(major_cm, weights['hidden'][0]), biases['hidden'][0])))
    hidden_layers.append(tf.nn.softplus(tf.add(tf.matmul(major_inputs, weights['hidden'][0]), biases['hidden'][0])))
    # Calculate the remaining hidden layers
    for i in range(1, len(hidden_layer_sizes)):
        prev_layer = hidden_layers[i-1]
        layer = tf.nn.softplus(tf.add(tf.matmul(prev_layer, weights['hidden'][i]), biases['hidden'][i]))
        hidden_layers.append(layer)
    out_layer = tf.matmul(hidden_layers[-1], weights['out']) + biases['out']
    logprobs_major = tf.nn.log_softmax(out_layer)

    # Minor
    hidden_layers = []
    # Assume there is at least one hidden layer...
    hidden_layers.append(tf.nn.softplus(tf.add(tf.matmul(minor_inputs, weights['hidden'][0]), biases['hidden'][0])))
    # Calculate the remaining hidden layers
    for i in range(1, len(hidden_layer_sizes)):
        prev_layer = hidden_layers[i-1]
        layer = tf.nn.softplus(tf.add(tf.matmul(prev_layer, weights['hidden'][i]), biases['hidden'][i]))
        hidden_layers.append(layer)
    out_layer = tf.matmul(hidden_layers[-1], weights['out']) + biases['out']
    logprobs_minor = tf.nn.log_softmax(out_layer)

    return params, major_inputs, minor_inputs, logprobs_major, logprobs_minor


def get_ll_from_nn(logprobs_major, logprobs_minor, logf, log1mf, lpf, counts):
    a_term = tf.add(
            tf.expand_dims(tf.expand_dims(logf, axis = -1), axis = -1),
            tf.expand_dims(logprobs_minor, axis = 0)
            )
    b_term = tf.add(
            tf.expand_dims(tf.expand_dims(log1mf, axis = -1), axis = -1),
            tf.expand_dims(logprobs_major, axis = 0)
            )
    tmp = tf.stack((a_term, b_term))
    lse = tf.reduce_logsumexp(tf.stack((a_term, b_term)), axis = 0)
    tmp = tf.multiply(lse, counts)
    tmp = tf.reduce_sum(tmp, axis = [1,2])
    tmp = lpf + tmp
    ll = tf.reduce_logsumexp(tmp)
    return ll


def get_ll_and_grads_tf(n_input, hidden_layer_sizes, num_f):
    params, major_inputs, minor_inputs, lpM, lpm = make_neural_net(n_input, hidden_layer_sizes)
    logf = tf.placeholder(tf.float64, [num_f])
    logpf = tf.placeholder(tf.float64, [num_f])
    log1mf = tf.placeholder(tf.float64, [num_f])
    counts = tf.placeholder(tf.float64, [None, 4])
    ll = get_ll_from_nn(lpM, lpm, logf, log1mf,logpf, counts) 
    grads = tf.gradients(ll, params)
    return params, major_inputs, minor_inputs, counts, logf, log1mf, logpf, ll, grads


def loglike_and_gradient_wrapper(params, cm, lo, maj, mino, num_pf_params, logf, log1mf, freqs, windows, ll_aux, sess):
    '''
    params are current parameter values
    cm is the localcm
    lo is tuple of tuple of direction-locobs
    maj, mino are major and minor bases
    ll_aux is all the variables that return from get_ll_tf
    num_pf_params, freqs, windows for lpf
    '''
    # Expand ll_aux
    params_tf, major_inputs, minor_inputs, counts_tf, logf_tf, log1mf_tf, logpf_tf, ll_tf, grads_tf = ll_aux

    # Get the AFD log-probability distribution
    lpf_np = bws.get_lpf(params[:num_pf_params], freqs, windows)

    if mino == 'N':

        # Have to average the likelihood over all the potential minor alleles
        alt_minors = [base for base in 'ACGT' if base != maj]

        gradients = []
        lls = []
        for alt_minor in alt_minors:
            major_cm, minor_cm, all_los = nn.get_major_minor_cm_and_los(cm, lo, maj, alt_minor)
            feed_dict = {
                    params_tf:params[num_pf_params:],
                    major_inputs:major_cm.astype(np.float64),
                    minor_inputs:minor_cm.astype(np.float64),
                    counts_tf: all_los[:,1:].astype(np.float64),
                    logf_tf: logf,
                    log1mf_tf: log1mf,
                    logpf_tf: lpf_np
                    }
            tll, tgrad = sess.run([ll_tf, grads_tf], feed_dict=feed_dict)
            lls.append(tll)
            gradients.append(tgrad[0])
        lls = np.array(lls)
        gradients = np.array(gradients)
        sign_grads = np.sign(gradients)
        logabs_grads = np.log(np.abs(gradients))
        logdenom = logsumexp(lls)
        pos_log_grads = (logabs_grads+lls[:,np.newaxis])
        neg_log_grads = (logabs_grads+lls[:,np.newaxis])
        pos_log_grads[sign_grads < 0] = -np.inf
        neg_log_grads[sign_grads >= 0] = -np.inf
        nngrads = np.exp(logsumexp(pos_log_grads, axis = 0)-logdenom) - np.exp(logsumexp(neg_log_grads, axis = 0)-logdenom)
        ll = logdenom - np.log(3.0)   # divided by 3 for average
    else:
        major_cm, minor_cm, all_los = nn.get_major_minor_cm_and_los(cm, lo, maj, mino)
        feed_dict = {
                params_tf:params[num_pf_params:],
                major_inputs:major_cm.astype(np.float64),
                minor_inputs:minor_cm.astype(np.float64),
                counts_tf: all_los[:,1:].astype(np.float64),
                logf_tf: logf,
                log1mf_tf: log1mf,
                logpf_tf: lpf_np
                }
        ll, nngrads = sess.run([ll_tf, grads_tf], feed_dict=feed_dict)
        nngrads = nngrads[0]
    grads = np.zeros_like(params)
    grads[num_pf_params:] = nngrads
    return ll, grads


def main1():
    # Import covariate data
    #fin = h5py.File('TR21_context_2_rb_20_localcm_small.h5')
    fin = h5py.File('sims_TR3_context_2_rb_20_localcm_more_heterplasmy.h5')
    bam = fin['locus_observations']['chrM'].keys()[0]
    lo = fin['locus_observations']['chrM'][bam]['100']
    lo = ((lo['f1'][:], lo['f2'][:]), (lo['r1'][:], lo['r2'][:]))
    cm = fin['covariate_matrices']['chrM'][bam]['100'][:]
    maj, mino = 'G', 'T'
    num_obs = cm.shape[0]
    major_cm, minor_cm, all_los = nn.get_major_minor_cm_and_los(cm,lo, maj, 'T')

    num_inputs = major_cm.shape[1]

    hidden_layer_sizes = [50,50]
    num_f = 100

    params_tf, major_inputs, minor_inputs, counts, logf, log1mf, logpf, ll, grads = get_ll_and_grads_tf(num_obs,
            num_inputs, hidden_layer_sizes, num_f)
    total_num_params = int(params_tf.shape[0])
    init = tf.global_variables_initializer()

    ll_target = -1.0*ll
    grads_target = tf.gradients(ll_target, params_tf)

    par_vals = npr.normal(size = total_num_params, scale = 0.05)
    freqs = bws.get_freqs(num_f)
    windows = bws.get_window_boundaries(num_f)
    lf = np.log(freqs)
    l1mf = np.log(1-freqs)
    pf_params = (-3, 5, 6)
    lpf_v = bws.get_lpf(pf_params, freqs, windows)

    with tf.Session() as sess:
        start = time.time()
        sess.run(init)
        res = sess.run([ll_target, grads_target], feed_dict = {params_tf: par_vals, logf: lf,
            log1mf: l1mf, logpf: lpf_v, counts: all_los[:,1:], major_inputs: major_cm,
            minor_inputs: minor_cm})
        dur = time.time() - start
        ll1, grads = res
        eps = 1e-5
        which = 200
        par_vals[which] += eps
        ll2 = sess.run(ll_target, feed_dict = {params_tf: par_vals, logf: lf,
            log1mf: l1mf, logpf: lpf_v, counts: all_los[:,1:].astype(np.float64),
            major_inputs: major_cm.astype(np.float64), minor_inputs: minor_cm.astype(np.float64)})
    n_grad = (ll2-ll1)/eps
    grad = res[1][0][which]
    print(n_grad, grad, dur)

def main2():
    # Import covariate data
    fin = h5py.File('sims_TR3_context_2_rb_20_localcm_more_heterplasmy.h5')
    bam = fin['locus_observations']['chrM'].keys()[0]
    lo = fin['locus_observations']['chrM'][bam]['100']
    lo = ((lo['f1'][:], lo['f2'][:]), (lo['r1'][:], lo['r2'][:]))
    cm = fin['covariate_matrices']['chrM'][bam]['100'][:]
    maj, mino = 'G', 'T'
    num_obs = cm.shape[0]
    major_cm, minor_cm, all_los = nn.get_major_minor_cm_and_los(cm,lo, maj, 'T')

    num_inputs = major_cm.shape[1]

    hidden_layer_sizes = [50,50]
    num_f = 100

    # Get the 
    ll_aux = get_ll_and_grads_tf(num_inputs, hidden_layer_sizes, num_f)

    total_num_params = int(ll_aux[0].shape[0])
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    # Get the initial values
    nn_par_vals = npr.normal(size = total_num_params, scale = 0.05)
    freqs = bws.get_freqs(num_f)
    windows = bws.get_window_boundaries(num_f)
    lf = np.log(freqs)
    l1mf = np.log(1-freqs)
    pf_params = (-3, 5, 6)
    par_vals = np.concatenate((pf_params, nn_par_vals))
    start = time.time()
    ll, grads = loglike_and_gradient_wrapper(par_vals, cm, lo, maj, mino, 3, lf, l1mf, freqs, windows, ll_aux, sess)
    dur1 = time.time()-start
    start = time.time()
    par_vals[8] += 0.2
    ll, grads = loglike_and_gradient_wrapper(par_vals, cm, lo, maj, mino, 3, lf, l1mf, freqs, windows, ll_aux, sess)
    dur2 = time.time()-start
    print('duration:', dur1, dur2)
    import pdb; pdb.set_trace()
    sess.close()


if __name__ == '__main__':
    main2()

