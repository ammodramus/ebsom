from __future__ import division, print_function
import re
import time   # for debugging / profiling

import numpy as np
import numpy.random as npr
import tensorflow as tf
import tables
from scipy.special import logsumexp

import beta_with_spikes_integrated as bws

np.seterr(divide='ignore')


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


def he_xavier_initialization(weight_shapes, num_cols_const):
    total_num_weights = (sum([ws[0]*ws[1] for ws in weight_shapes])
                         + (num_cols_const+4)*weight_shapes[0][1])

    init_vals = np.zeros(total_num_weights, dtype=np.float32)

    weight_shapes[0] = list(weight_shapes[0])
    weight_shapes[0][0] += num_cols_const + 4

    start = 0
    for ws in weight_shapes:
        num_weights = ws[0]*ws[1]
        end = start + num_weights
        init_vals[start:end] = npr.randn(num_weights) * 2.0/np.sqrt(ws[0])
        start = end

    return init_vals

def make_neural_net(n_cols, n_cols_const, hidden_layer_sizes, num_pf_params):
    if len(hidden_layer_sizes) < 1:
        raise ValueError('hidden_layer_sizes must contain at least one integer')

    n_bases = 4

    forward_inputs = tf.placeholder(tf.float32, (None, n_cols),
                                    name='for_inputs')
    reverse_inputs = tf.placeholder(tf.float32, (None, n_cols),
                                    name='rev_inputs')
    
    # note:  const input will have both the constant covariates and the "true" base
    major_forward_inputs_const = tf.placeholder(
        tf.float32, (1,n_cols_const+n_bases), name='maj_for_inputs_const')
    minor_forward_inputs_const = tf.placeholder(
        tf.float32, (1,n_cols_const+n_bases), name='min_for_inputs_const')
    major_reverse_inputs_const = tf.placeholder(
        tf.float32, (1,n_cols_const+n_bases), name='maj_rev_inputs_const')
    minor_reverse_inputs_const = tf.placeholder(
        tf.float32, (1,n_cols_const+n_bases), name='min_rev_inputs_const')

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

    # Create parameters tensor containing all of the weights and biases as a
    # single placeholder array

    init_weight_vals = he_xavier_initialization(weight_shapes, n_cols_const)
    init_bias_vals = np.zeros(sum(bias_shapes), dtype=np.float32)
    init_pf_params = (0.5,2,8)
    init_param_vals = np.concatenate((init_pf_params, init_weight_vals,
                                       init_bias_vals)).astype(np.float32)

    with tf.name_scope('params'):
        params = tf.Variable(init_param_vals, name = 'params',
                             dtype=tf.float32)

    # Define weights and biases in terms of indices to this parameters tensor.
    # This replaces the weight- and bias-creation below.
    weights = {}
    weights['hidden'] = []

    start = num_pf_params   # the P(f) params come first

    end = n_cols*hidden_layer_sizes[0] + start
    weights['hidden'].append(
        tf.reshape(params[start:end], [n_cols, hidden_layer_sizes[0]]))
    start = end
    
    end = start + hidden_layer_sizes[0]*(n_cols_const+n_bases)
    weights['const'] = tf.reshape(
        params[start:end], (n_cols_const+n_bases, hidden_layer_sizes[0]))
    start = end

    for i in range(1, len(hidden_layer_sizes)):
        end = hidden_layer_sizes[i-1]*hidden_layer_sizes[i] + start
        weights['hidden'].append(
            tf.reshape(params[start:end], 
                       (hidden_layer_sizes[i-1],hidden_layer_sizes[i]))
        )
        start = end
    end = start + hidden_layer_sizes[-1]*n_bases
    weights['out'] = tf.reshape(
        params[start:end], (hidden_layer_sizes[-1], n_bases))
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
    
    assert start == total_num_params + num_pf_params, 'start: {}, total_num_params: {}'.format(start, total_num_params)

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
        
        input0 = tf.concat((major_forward_input0, major_reverse_input0),
                            axis=0) + biases['hidden'][0]
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


def get_ll_gradient_and_inputs(n_cols, n_cols_const, n_bams,
                               hidden_layer_sizes, nfreqs, num_pf_params):
    n_cols_const += n_bams
    ll_aux = make_neural_net(n_cols, n_cols_const, hidden_layer_sizes,
                             num_pf_params)
    (params, forward_inputs, reverse_inputs, major_forward_inputs_const,
     minor_forward_inputs_const, major_reverse_inputs_const,
     minor_reverse_inputs_const, logprobs_major, logprobs_minor) = ll_aux
    logf = tf.placeholder(tf.float32, [nfreqs], name = 'logf')
    log1mf = tf.placeholder(tf.float32, [nfreqs], name = 'log1mf')
    logpf = tf.placeholder(tf.float32, [nfreqs], name = 'logpf')
    counts = tf.placeholder(tf.float32, [None, 4], name = 'counts')
    a_term = tf.add(
            tf.expand_dims(tf.expand_dims(logf, axis=-1), axis=-1),
            tf.expand_dims(logprobs_minor, axis = 0)
            )
    b_term = tf.add(
            tf.expand_dims(tf.expand_dims(log1mf, axis=-1), axis=-1),
            tf.expand_dims(logprobs_major, axis=0)
            )
    lse = tf.reduce_logsumexp(tf.stack((a_term, b_term)), axis=0)
    tmp = tf.multiply(lse, counts)
    b_sums = tf.reduce_sum(tmp, axis=[1,2], name='freq_logposteriors')
    f_posts = logpf + b_sums
    ll = tf.reduce_logsumexp(f_posts, name='loglike')
    nn_grads = tf.gradients(ll, params, name='nn_grads')
    return (ll, nn_grads, params, forward_inputs, reverse_inputs,
            major_forward_inputs_const, minor_forward_inputs_const,
            major_reverse_inputs_const,  minor_reverse_inputs_const, counts,
            logf, log1mf, logpf, b_sums)



def loglike_and_gradient_wrapper(forward_cov, reverse_cov,
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
    (ll_tf, nn_grads_tf, nn_params_tf, forward_inputs_tf, reverse_inputs_tf,
     major_forward_inputs_const_tf, minor_forward_inputs_const_tf,
     major_reverse_inputs_const_tf,  minor_reverse_inputs_const_tf, counts_tf,
     logf_tf, log1mf_tf, logpf_tf, b_sums_tf) = ll_aux

    params = sess.run(ll_aux[2])
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
            tll, tgrad = loglike_and_gradient_wrapper(forward_cov,
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
        grads = (np.exp(logsumexp(pos_log_grads, axis=0)-logdenom)
                 - np.exp(logsumexp(neg_log_grads, axis=0)-logdenom))
        ll = logdenom - np.log(3.0)   # divided by 3 for average
        return ll, grads

    else:
        (major_forward_base_cov, minor_forward_base_cov,
         major_reverse_base_cov, minor_reverse_base_cov) = (
             get_base_covs(major, minor))

        major_forward_const = np.concatenate(
            (major_forward_base_cov, forward_const_cov))[np.newaxis,:]
        minor_forward_const = np.concatenate(
            (minor_forward_base_cov, forward_const_cov))[np.newaxis,:]
        major_reverse_const = np.concatenate(
            (major_reverse_base_cov, reverse_const_cov))[np.newaxis,:]
        minor_reverse_const = np.concatenate(
            (minor_reverse_base_cov, reverse_const_cov))[np.newaxis,:]

        combined_counts = np.concatenate((counts_forward, counts_reverse),
                                         axis=0)

        feed_dict = {
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
        ll, all_grads, b_sums = sess.run([ll_tf, nn_grads_tf, b_sums_tf],
                                       feed_dict=feed_dict)
        all_grads = all_grads[0]
        grads = np.zeros_like(params)
        grads[:] = all_grads

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
                pos_log = (logsumexp(np.log(np.abs(dpfs[filt]))
                                     + b_sums[filt]) - ll)
            else:
                pos_log = -np.inf
            if np.any(~filt):
                neg_log = (logsumexp(np.log(np.abs(dpfs[~filt])) +
                                     b_sums[~filt]) - ll)
            else:
                neg_log = -np.inf
            grads[i] = np.exp(pos_log) - np.exp(neg_log)

        return ll, grads


class ErrorModel(object):
    def __init__(self, processed_data, hidden_layer_sizes, nfreqs, conc_factor,
                 num_pf_params, session):
        self.data = processed_data
        self.metadata = processed_data.root.meta.metadata
        self.metadata_np = self.metadata.read()
        self.meta_cols = {el: i for i, el in
                          enumerate(processed_data.root.meta.metadata.colnames)
                          }
        bam_idx = self.meta_cols['bam']
        ref_idx = self.meta_cols['reference']
        position_idx = self.meta_cols['position']
        self.row_dict = {}
        for i, row in enumerate(self.metadata_np):
            bam = row[bam_idx]
            ref = row[ref_idx]
            pos = row[position_idx]
            key = (bam,ref,pos)
            self.row_dict[key] = i
        self.for_cov = processed_data.root.data.for_cov
        self.rev_cov = processed_data.root.data.rev_cov
        self.for_obs = processed_data.root.data.for_obs
        self.rev_obs = processed_data.root.data.rev_obs
        self.ncols = self.for_cov.shape[1]
        _fcc_idx = self.meta_cols['forward_const_cov']
        self.ncols_const = self.metadata_np[0][_fcc_idx].shape[0]
        self.hidden_layer_sizes = hidden_layer_sizes
        self.nfreqs = nfreqs
        self.conc_factor = conc_factor
        self.num_pf_params = num_pf_params

        self.bam_counts = self.metadata.attrs.bam_counts
        self.num_bams = len(self.bam_counts)
        bam_means = self.bam_counts / np.sum(self.bam_counts)
        bam_stds = np.sqrt(bam_means*(1-bam_means))
        # these to be used for expanding the one-hot bam encoding
        self.bam_mins = -bam_means/bam_stds
        self.bam_maxes = (1-bam_means)/bam_stds

        self.ll_aux = get_ll_gradient_and_inputs(self.ncols, self.ncols_const,
                                                 self.num_bams,
                                                 hidden_layer_sizes, nfreqs,
                                                 num_pf_params)
        self.freqs = bws.get_freqs(nfreqs, conc_factor)
        self.nparams = self.num_pf_params + int(self.ll_aux[2].shape[0])
        with np.errstate(divide='ignore'):
            self.logf = np.log(self.freqs)
            self.log1mf = np.log(1-self.freqs)
        self.windows = bws.get_window_boundaries(nfreqs, conc_factor)
        self.sess = session
        self.bam_enum_values = {key:val for key, val in
                                self.metadata.get_enum('bam')}
        self.ref_enum_values = {key:val for key, val in
                                self.metadata.get_enum('reference')}
        self.reverse_bam_enum_values = {val:key for key, val in
                                self.metadata.get_enum('bam')}
        self.reverse_ref_enum_values = {val:key for key, val in
                                self.metadata.get_enum('reference')}


        # profiling
        self.time_retrieving_data = 0.0
        self.time_calculating_grads = 0.0
        
    def loglike_and_gradient(self, bam, ref, pos):
        start = time.time()

        bam_idx = self.bam_enum_values[bam]
        ref_idx = self.ref_enum_values[ref]
        key = (bam_idx, ref_idx, pos)
        meta_row = self.metadata_np[self.row_dict[key]]
        forward_start = meta_row[self.meta_cols['forward_start']]
        forward_end = meta_row[self.meta_cols['forward_end']]
        reverse_start = meta_row[self.meta_cols['reverse_start']]
        reverse_end = meta_row[self.meta_cols['reverse_end']]
        forward_data = self.for_cov[forward_start:forward_end]
        reverse_data = self.rev_cov[reverse_start:reverse_end]
        forward_obs = self.for_obs[forward_start:forward_end]
        reverse_obs = self.rev_obs[reverse_start:reverse_end]
        forward_const_no_bam = meta_row[self.meta_cols['forward_const_cov']]
        reverse_const_no_bam = meta_row[self.meta_cols['reverse_const_cov']]

        bam_cov = self.bam_mins.copy()
        bam_cov[bam_idx] = self.bam_maxes[bam_idx]
        forward_const = np.concatenate((forward_const_no_bam, bam_cov))
        reverse_const = np.concatenate((reverse_const_no_bam, bam_cov))

        major = meta_row[self.meta_cols['major']]
        minor = meta_row[self.meta_cols['minor']]

        dur = time.time()-start
        self.time_retrieving_data += dur

        start = time.time()
        val = loglike_and_gradient_wrapper(forward_data, reverse_data,
                                            forward_const, reverse_const,
                                            forward_obs, reverse_obs, major,
                                            minor, self.num_pf_params,
                                            self.logf, self.log1mf, self.freqs,
                                            self.windows, self.ll_aux,
                                            self.sess)
        dur = time.time()-start
        self.time_calculating_grads += dur

        return val


if __name__ == '__main__':

    import numpy.random as npr

    dat = tables.File('test.h5')

    x = ErrorModel(processed_data=dat, hidden_layer_sizes=[50,20], nfreqs=150,
                   conc_factor=20, num_pf_params=3)


    params = npr.normal(size=x.nparams)/1000
    print(x.loglike_and_gradient(params, 'srt.m468c4_S23.rd.nvcReady.bam',
                                 'chrM', 4000))

    dat.close()
