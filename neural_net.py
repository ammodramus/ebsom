from __future__ import print_function
import numpy as np
import numpy.random as npr
import h5py
import numexpr as ne

from numba import jit
from scipy.special import logsumexp
from itertools import izip


import beta_with_spikes_integrated as bws
import cyutil as cut

def relu(a):
    return np.maximum(0, a)
def d_relu(a):
    return (a >= 0).astype(np.float64)

def softplus(a):
    return np.log(1+np.exp(a))
def d_softplus(a):
    return 1/(1+np.exp(-a))

activ = softplus
d_activ = d_softplus

def softmax(a):
    expa = np.exp(a)
    return expa / np.sum(expa, axis = 0)

# just for a single observation
def d_softmax(probs):
    # symmetric matrix, so order doesn't matter.
    out = -1*np.outer(probs, probs)
    out[np.diag_indices_from(out)] += probs
    return out

@jit
def broadcast_d_softmax(logprobs):
    '''
    probs is (4,n) matrix, n is num observations
    
    out is a (4,n,4) matrix, with n is the index of the observation
    out[i,j,k] gives output derivative wrt the i'th variable,
               the j'th observation, for the the k'th probability
    '''
    nobs = probs.shape[1]
    out = np.zeros((4, nobs, probs.shape[0]))
    for j in range(nobs):
        out[:,j,:] = d_softmax(probs[:,j])
    return out

def logsoftmax(logprobs):
    return logprobs - logsumexp(logprobs, axis = 0)

def d_logsoftmax(logprobs):
    '''
    logprobs is (4,)-array of log-probabilities
    '''
    out = np.zeros((4,4))
    out[:,:] = -np.exp(logprobs[np.newaxis,:])
    out[np.diag_indices_from(out)] += 1
    return out

@jit
def broadcast_d_logsoftmax(logprobs):
    '''
    logprobs is (4,n) matrix, n is num observations
    
    out is a (4,n,4) matrix, with n is the index of the observation
    out[i,j,k] gives output derivative wrt the i'th variable,
               the j'th observation, for the the k'th probability
    '''
    nobs = logprobs.shape[1]
    out = np.zeros((logprobs.shape[0], nobs, logprobs.shape[0]))
    for j in range(nobs):
        out[:,j,:] = d_logsoftmax(logprobs[:,j])
    return out


def get_major_minor_cm_and_los(cm, lo, major, minor):
    major_cms = []
    minor_cms = []
    los = []
    for direc_idx in [0,1]:
        for rn_idx in [0,1]:
            tlo = lo[direc_idx][rn_idx]
            if tlo.shape[0] == 0 or tlo.shape[1] == 0:
                continue
            tcm = cm[tlo[:,0]]
            readtwos = np.ones(tlo.shape[0]) * rn_idx
            direc_major = str(major) if direc_idx == 0 else cut.comp(str(major))
            direc_minor = str(minor) if direc_idx == 0 else cut.comp(str(minor))
            major_col_idx = 'ACGT'.index(direc_major)
            major_base_columns = np.zeros((tcm.shape[0], 4))
            major_base_columns[major_col_idx] = 1.0
            minor_col_idx = 'ACGT'.index(direc_minor)
            minor_base_columns = np.zeros((tcm.shape[0], 4))
            minor_base_columns[minor_col_idx] = 1.0
            major_cm = np.column_stack((major_base_columns, readtwos[:,np.newaxis], tcm))
            major_cms.append(major_cm)
            minor_cm = np.column_stack((minor_base_columns, readtwos[:,np.newaxis], tcm))
            minor_cms.append(minor_cm)
            los.append(tlo)
    major_cm = np.row_stack(major_cms)
    minor_cm = np.row_stack(minor_cms)
    all_los = np.row_stack(los)
    return major_cm, minor_cm, all_los
'''
def get_major_minor_cm_and_los(cm, lo, major, minor):
    major_cms = []
    minor_cms = []
    los = []
    for direc in 'fr':
        for rn in '12':
            key = direc + rn
            tlo = lo[key][:]
            tcm = cm[tlo[:,0]]
            readtwos = np.ones(tlo.shape[0]) * (int(rn)-1)
            direc_major = str(major) if direc == 'f' else cut.comp(str(major))
            direc_minor = str(minor) if direc == 'f' else cut.comp(str(minor))
            major_col_idx = 'ACGT'.index(direc_major)
            major_base_columns = np.zeros((tcm.shape[0], 4))
            major_base_columns[major_col_idx] = 1.0
            minor_col_idx = 'ACGT'.index(direc_minor)
            minor_base_columns = np.zeros((tcm.shape[0], 4))
            minor_base_columns[minor_col_idx] = 1.0
            major_cm = np.column_stack((major_base_columns, readtwos[:,np.newaxis], tcm))
            major_cms.append(major_cm)
            minor_cm = np.column_stack((minor_base_columns, readtwos[:,np.newaxis], tcm))
            minor_cms.append(minor_cm)
            los.append(tlo)
    major_cm = np.row_stack(major_cms)
    minor_cm = np.row_stack(minor_cms)
    all_los = np.row_stack(los)
    return major_cm, minor_cm, all_los
'''


# get distribution frequencies and windows
def get_freqs_windows_lf_l1mf(num_f):
    freqs = bws.get_freqs(num_f)
    windows = bws.get_window_boundaries(num_f)
    lf = np.log(freqs)
    l1mf = np.log(1-freqs)
    return freqs, windows, lf, l1mf


def initialize_matrices(num_inputs, hidden_layer_sizes, num_obs):
    num_outputs = 4
    weights = [None]
    biases = [None]
    activations = []
    deltas = []
    weighted_inputs = []
    deriv_activations = []
    num_neurons = [num_inputs] + hidden_layer_sizes + [num_outputs]
    for a, b in izip(num_neurons[:-1], num_neurons[1:]):
        # notice that b and a are switched
        weights.append(np.zeros((b,a)))
    for i, n in enumerate(num_neurons):
        if i > 0:
            biases.append(np.zeros(n))
        activations.append(np.zeros((n, num_obs), order = 'C'))
        deltas.append(np.zeros((4, n, num_obs), order = 'C'))
        weighted_inputs.append(np.zeros((n, num_obs), order = 'C'))
        deriv_activations.append(np.zeros((n, num_obs), order = 'C'))
    return weights, biases, weighted_inputs, activations, deriv_activations, deltas

def get_num_params(num_inputs, hidden_layer_sizes, num_outputs):
    n_params = 0
    num_neurons = [num_inputs] + hidden_layer_sizes + [num_outputs]
    for a, b in izip(num_neurons[:-1], num_neurons[1:]):
        # notice that b and a are switched
        #weights.append(npr.uniform(-0.1, 0.1, size = (b,a)))
        n_params += b*a
    n_params += sum(num_neurons[1:])
    return n_params


def neural_net_logprobs_and_gradients(inputs, weights, biases,
                                      weighted_inputs, activations,
                                      deriv_activations, deltas):
    num_layers = len(biases)-1
    activations[0] = inputs
    for i in range(num_layers):
        np.dot(weights[i+1], activations[i], out = weighted_inputs[i+1])
        weighted_inputs[i+1] += biases[i+1][:,np.newaxis]
        activations[i+1] = activ(weighted_inputs[i+1])
        deriv_activations[i+1] = d_activ(weighted_inputs[i+1])
    output = logsoftmax(activations[-1])

    deriv_output = broadcast_d_logsoftmax(output)
    for which_respect in range(4):
        deltas[-1][which_respect] = deriv_output.T[:,:,which_respect] * d_activ(weighted_inputs[-1])
        #deltas[-1][which_respect] = deriv_output[:,:,which_respect] * d_activ(weighted_inputs[-1])
        for i in range(len(deltas)-2, 0, -1):
            deltas[i][which_respect,:] = np.dot(weights[i+1].T, deltas[i+1][which_respect]) * deriv_activations[i]
    # deltas[i][j][k][l] gives derivative of the j'th logprob wrt weighted input
    #        in the i'th layer, k'th neuron in the layer, l'th observation
    return output, deltas

def neural_net_logprobs(inputs, weights, biases, weighted_inputs, activations):
    num_layers = len(biases)-1
    activations[0] = inputs
    for i in range(num_layers):
        np.dot(weights[i+1], activations[i], out = weighted_inputs[i+1])
        weighted_inputs[i+1] += biases[i+1][:,np.newaxis]
        activations[i+1] = activ(weighted_inputs[i+1])
    output = logsoftmax(activations[-1])

    return output


def neural_net_logprobs_wrapper(params, inputs, matrices):
    weights, biases, weighted_inputs, activations  = matrices[:-2]

    start = 0
    for idx, w in enumerate(weights[1:]):
        wshp = w.shape
        end = start + wshp[0]*wshp[1]
        w[:,:] = params[start:end].reshape(wshp, order = 'C')
        start = end 
    for b in biases[1:]:
        bshp = b.shape
        end = start + bshp[0]
        b[:] = params[start:end].reshape(bshp, order = 'C')
        start = end
    assert end == params.shape[0], 'end is {}, params.shape[0] is {}'.format(end, params.shape[0])
    logprobs = neural_net_logprobs(inputs, weights, biases,
            weighted_inputs, activations)
    return logprobs


def neural_net_logprobs_and_gradients_wrapper(params, inputs, matrices):
    weights, biases, weighted_inputs, activations, deriv_activations, deltas = matrices
    # fill in the weights and biases, then evaluate
    start = 0
    for idx, w in enumerate(weights[1:]):
        wshp = w.shape
        end = start + wshp[0]*wshp[1]
        w[:,:] = params[start:end].reshape(wshp, order = 'C')
        start = end 
    for b in biases[1:]:
        bshp = b.shape
        end = start + bshp[0]
        b[:] = params[start:end].reshape(bshp, order = 'C')
        start = end
    assert end == params.shape[0], 'end is {}, params.shape[0] is {}'.format(end, params.shape[0])
    logprobs, deltas = neural_net_logprobs_and_gradients(inputs, weights, biases,
                                             weighted_inputs, activations,
                                             deriv_activations, deltas)
    # get gradients from deltas
    n_obs = deltas[1].shape[-1]
    grads = np.zeros((params.shape[0], 4, n_obs))
    start = 0
    for d, a in izip(deltas[1:], activations):
        for j in range(4):
            for l in range(n_obs):
                # probably want to jit these loops
                # also, reduce number of []'s
                #print a[:,l].shape, d[j,:,l].shape, d.shape
                w_prime = np.ravel(np.outer(d[j,:,l], a[:,l]), order = 'C')
                end = start + w_prime.shape[0]
                grads[start:end,j,l] = w_prime
        start = end
    
    for d in deltas[1:]:
        for j in range(4):
            end = start + d[j,:,:].shape[0]
            grads[start:end, j, :] = d[j,:,:]
        start = end
    assert start == params.shape[0]
    return logprobs, grads


# set up the neural net
def set_up_neural_net(num_inputs, hidden_layer_sizes, num_obs):
    '''
    parameters
    ---------------
    num_inputs: number of input variables (columns in cm) [int]
    hidden_layer_sizes: number of nodes in each hidden layer [list of ints]
    
    returns
    ---------------
    matrices, num_params
    
    where matrices is (weights, biases, weighted_inputs, activations, deriv_activations, deltas)
    and num_params is the number of weights and biases that are parameters in the neural net
         (num_params does not include AFS distribution parameters)
    '''
    num_outputs = 4
    matrices = initialize_matrices(num_inputs, hidden_layer_sizes, num_obs)
    num_params = get_num_params(num_inputs, hidden_layer_sizes, num_outputs)
    return matrices, num_params

#@jit
def get_a_sums_ne(d_prob_minor, d_prob_major, logprobs_minor, logprobs_major, fs, logfs, log1mfs, locobs):
    # d_prob_minor.shape is (num_nn_params, 4, num_obs)
    # d_prob_major.shape is (num_nn_params, 4, num_obs)
    # logprobs_minor.shape is (4, num_obs)
    # logprobs_major.shape is (4, num_obs)
    # locobs.shape is (num_obs, 5)
    # fs.shape is (num_fs,)
    
    num_nn_params = d_prob_minor.shape[0]
    num_obs = d_prob_minor.shape[2]
    num_fs = fs.shape[0]
    
    out = np.zeros((num_nn_params, num_fs))
    
    assert locobs.shape[0] == d_prob_minor.shape[2]
    probs_minor = np.exp(logprobs_minor).astype(np.float32)
    probs_major = np.exp(logprobs_major).astype(np.float32)
    d_prob_minor = d_prob_minor.astype(np.float32)
    d_prob_major = d_prob_major.astype(np.float32)
    fs = fs.astype(np.float32)
    counts = locobs[:,1:].T[np.newaxis,:,:].astype(np.float32)
    for l in range(num_fs):
        #print('\r{}'.format(l), end = '')
        f = fs[l]
        denom = (f*probs_minor + (1-f)*probs_major)[np.newaxis,:,:]
        summands = ne.evaluate('counts*(f*d_prob_minor+(1-f)*d_prob_major)/denom', global_dict = {},
                          local_dict = {'counts':counts, 'f':f, 'd_prob_minor':d_prob_minor,
                                        'd_prob_major':d_prob_major,'denom': denom}, truediv = True)
        out[:,l] = summands.sum(axis = (1,2))
    return out



#@jit
def get_pos_neg_logabs_numerators(logabs_a_sums, sign_a_sums, b_sums):
    '''
    logabs_a_sums[i,j] is log(abs(x)), where x[i,j] is the 'A' sum for the i'th parameter and j'th frequency
    same dimensions for sign_a_sums
    '''
    
    num_vars, num_fs = logabs_a_sums.shape
    pos_out = np.zeros(num_vars)
    neg_out = np.zeros(num_vars)
    for i in range(num_vars):
        is_nonneg = sign_a_sums[i,:] >= 0   # signs of a_sums for this variable
        is_neg = ~is_nonneg
        if np.any(is_nonneg):
            pos_out[i] = logsumexp(b_sums[is_nonneg]+logabs_a_sums[i,is_nonneg])
        else:
            pos_out[i] = -np.inf
        if np.any(is_neg):
            neg_out[i] = logsumexp(b_sums[is_neg]+logabs_a_sums[i,is_neg])
        else:
            neg_out[i] = -np.inf
    return pos_out, neg_out


#@jit
def get_b_sums(logfs, log1mfs, logprobs_minor, logprobs_major, los):
    counts = los[:,1:].T.astype(np.float32)   # maybe move casting outside of function
    num_fs = logfs.shape[0]
    out = np.zeros(num_fs)
    buf = np.zeros_like(logprobs_minor)
    for i in range(num_fs):
        logf = logfs[i]
        log1mf = log1mfs[i]
        np.logaddexp(logf+logprobs_minor, log1mf+logprobs_major, out = buf)
        buf *= counts
        out[i] = buf.sum()
    return out


def get_loglike_and_gradient(params, major_cm, minor_cm, all_los, matrices, num_pf_params, freqs, windows, major, minor, eps = 1e-6):
    num_fs = freqs.shape[0]
    logfs = np.log(freqs)
    log1mfs = np.log(1-freqs)
    pf_pars = params[:num_pf_params]
    nn_pars = params[num_pf_params:]
    lpf = bws.get_lpf(pf_pars, freqs, windows)
    logprobs_minor, d_logprob_minor = neural_net_logprobs_and_gradients_wrapper(nn_pars, minor_cm.T, matrices)
    logprobs_major, d_logprob_major = neural_net_logprobs_and_gradients_wrapper(nn_pars, major_cm.T, matrices)
    probs_minor = np.exp(logprobs_minor)
    probs_major = np.exp(logprobs_major)
    d_prob_minor = probs_minor[np.newaxis,:,:]*d_logprob_minor
    d_prob_major = probs_major[np.newaxis,:,:]*d_logprob_major
    a_sums = get_a_sums_ne(d_prob_minor, d_prob_major, logprobs_minor,
                           logprobs_major, freqs, logfs, log1mfs, all_los)
    logabs_a_sums_f = ne.evaluate('log(abs(x))', local_dict = {'x': a_sums})
    sign_a_sums_f = np.sign(a_sums, order = 'F')
    b_sums = get_b_sums(logfs, log1mfs, logprobs_minor, logprobs_major, all_los)
    logdenoms = lpf + b_sums
    logabs_num_pos, logabs_num_neg = get_pos_neg_logabs_numerators(logabs_a_sums_f, sign_a_sums_f, logdenoms)
    logdenom = logsumexp(logdenoms)  # also the log-likelihood
    logabs_num_pos -= logdenom
    logabs_num_neg -= logdenom
    
    ret = np.zeros_like(params)
    # calculate gradient for pf_params   MOVE THIS TO A SEPARATE FUNCTION
    pf = np.exp(lpf)
    for i in range(num_pf_params):
        pf_pars[i] += eps  # inc by eps
        pf2 = np.exp(bws.get_lpf(pf_pars,freqs,windows))
        pf_pars[i] -= eps  # fix the eps
        dpfs = (pf2-pf)/eps
        filt = dpfs >= 0
        pos_log = logsumexp(np.log(dpfs[filt]) + b_sums[filt]) - logdenom
        neg_log = logsumexp(ne.evaluate('log(abs(x))', local_dict = {'x': dpfs[~filt]}) + b_sums[~filt]) - logdenom
        ret[i] = np.exp(pos_log) - np.exp(neg_log)
    
    ret[num_pf_params:] = np.exp(logabs_num_pos)-np.exp(logabs_num_neg)
    
    return logdenom, ret

def get_gradient(params, cm, lo, matrices, num_pf_params, freqs, windows, major, minor, eps = 1e-6):
    # just a simple wrapper around get_loglike_and_gradient()
    # (have to calculate likelihood to calculate gradient)
    return get_loglike_and_gradient(params, cm, lo, matrices, num_pf_params, freqs, windows, major, minor, eps)[1]

def get_loglike(params, cm, lo, matrices, num_pf_params, freqs, windows, major, minor):
    if minor == 'N':
        alt_minors = [base for base in 'ACGT' if base != major]
        lls = []
        for alt_minor in alt_minors:
            lls.append(get_loglike(params, cm, lo, matrices, num_pf_params, freqs, windows, major, alt_minor))
        ll = logsumexp(lls) - np.log(3.0)
        return ll
    logfs = np.log(freqs)
    log1mfs = np.log(1-freqs)
    pf_pars = params[:num_pf_params]
    nn_pars = params[num_pf_params:]
    lpf = bws.get_lpf(pf_pars, freqs, windows)
    major_cm, minor_cm, all_los = get_major_minor_cm_and_los(cm,lo, major, minor)
    # don't need backpropagation here; TODO write another NN function for just logprobs
    logprobs_minor = neural_net_logprobs_wrapper(nn_pars, minor_cm.T, matrices)
    logprobs_major = neural_net_logprobs_wrapper(nn_pars, major_cm.T, matrices)
    logsummands = get_b_sums(logfs, log1mfs, logprobs_minor, logprobs_major, all_los)
    logsummands += lpf
    ll = logsumexp(logsummands)
    return ll

def loglike_and_gradient_wrapper(params, cm, lo, num_obs, maj, mino, hidden_layer_sizes, num_pf_params, freqs, windows):
    num_obs = cm.shape[0]
    if mino == 'N':
        alt_minors = [base for base in 'ACGT' if base != maj]
        gradients = []
        lls = []
        for alt_minor in alt_minors:
            major_cm, minor_cm, all_los = get_major_minor_cm_and_los(cm, lo, maj, alt_minor)
            num_inputs = major_cm.shape[1]
            matrices, num_params = set_up_neural_net(num_inputs, hidden_layer_sizes, num_obs)
            tll, tgrad = get_loglike_and_gradient(params, major_cm, minor_cm, all_los, matrices, num_pf_params, freqs, windows, maj, alt_minor)
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
    else:
        major_cm, minor_cm, all_los = get_major_minor_cm_and_los(cm, lo, maj, mino)
        num_inputs = major_cm.shape[1]
        matrices, num_params = set_up_neural_net(num_inputs, hidden_layer_sizes, num_obs)
        ll, grads = get_loglike_and_gradient(params, major_cm, minor_cm, all_los, matrices, num_pf_params, freqs, windows, maj, mino)
    return ll, grads

def loglike_and_gradient_arg_wrapper(args):
    return loglike_and_gradient_wrapper(*args)


if __name__ == '__main__':
    fin = h5py.File('TR21_context_2_rb_20_localcm_small.h5')
    bam = fin['locus_observations']['chrM'].keys()[0]
    lo = fin['locus_observations']['chrM'][bam]['0']
    cm = fin['covariate_matrices']['chrM'][bam]['0'][:]

    num_f = 100
    freqs = bws.get_freqs(num_f)
    windows = bws.get_window_boundaries(num_f)
    lf = np.log(freqs)
    l1mf = np.log(1-freqs)

    maj, mino = 'G', 'T'
    num_obs = cm.shape[0]
    x, _, __ = get_major_minor_cm_and_los(cm,lo, maj, 'T')
    num_inputs = x.shape[1]
    hidden_layer_sizes = [50]
    matrices, num_params = set_up_neural_net(num_inputs, hidden_layer_sizes, num_obs)
    npr.seed(1)
    init_params = npr.uniform(-0.1, 0.1, size = num_params)
    num_pf_params = 3
    pf_params = (-3, 5, 6)
    params = np.concatenate((pf_params, init_params))

    loglike, gradient = get_loglike_and_gradient(params, cm, lo, matrices, 3, freqs, windows, maj, mino)
    print('gradient.shape:', gradient.shape)

    eps = 1e-6
    params2 = params.copy()
    which = 10
    params2[which] += eps
    loglike2 = get_loglike(params2, cm, lo, matrices, 3, freqs, windows, maj, mino)

    print('numerical:', (loglike2-loglike)/eps)
    print('analytical:', gradient[which])
