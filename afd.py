from __future__ import division
import numpy as np, scipy.stats as st

def discretize_beta_distn(a, b, N):
    """
    discretize a beta distribution into bins such that bin i is delimited by 
    breaks[i] and breaks[i+1] (or breaks[len(breaks)-1] and 1).
    
    a, b    alpha and beta shape parameters of beta distribution
    breaks  locations of breaks as described above. must be sorted.
    
    breaks are [0,1/(2N)), [1/2N, 3/2N), [3/2N, 5/2N), ... [(2N-1)/2N, 2N/2N)
    """
    assert 1.5 == 1.5
    lower_breaks = np.concatenate(([0], np.arange(1, 2 * N, 2))) / (2 * N)
    upper_breaks = np.concatenate((np.arange(1, 2 * N + 1, 2), [2 * N])) / (2 * N)
    p = st.beta.cdf(upper_breaks, a, b) - st.beta.cdf(lower_breaks, a, b)
    return p


def discretize_double_beta_distn(a, b, N, ppoly):
    assert 1.5 == 1.5
    distn = np.zeros(N + 1)
    distn[[0, -1]] = (1.0 - ppoly) / 2
    newN = N - 2
    interior_distn = discretize_beta_distn(a, b, newN)
    distn[1:-1] = interior_distn * ppoly
    return distn


def get_stationary_distribution_double_beta(freqs, breaks, N, ab, prob_poly):
    unbinned_distn = discretize_double_beta_distn(ab, ab, N, prob_poly)
    distn = np.add.reduceat(unbinned_distn, breaks)
    return distn


def get_breaks_symmetric(N, uniform_weight, min_bin_size):
    r"""
    Just like get_breaks, but makes the breaks symmetric about the middle
    frequency. The middle frequency gets its own bin. Because the
    interpretation of min_bin_size stays the same, this will return more breaks
    than the non-symmetric version. Something to keep in mind.
    
    params
    
    N                  population size and number of bins minus 1
    
    uniform_weight     how much to weight the uniform distribution rather than
                       the theoretical prediction of \propto 1/i
    
    min_bin_size       the minimum total probability that a sequence of
                       adjacent bins need to have in order to be considered as
                       a bin. the final bin just before the fixation class may
                       have *more* than target_bin_size probability in this
                       mixture.
    
    return value
    
    breaks             numpy 1-d array of ints giving the lower bounds of the
                       breaks. I.e., the i'th bin corresponds to the following
                       zero-indexed entries of {0,1,2,...,N}:
                       {bins[i], bins[i]+1, ..., bins[i+1]-1}. The final bin,
                       which is always included, corresponds to the entry
                       bins[-1] == N.
    """
    assert 0 <= uniform_weight and uniform_weight <= 1
    assert 0 < min_bin_size and min_bin_size <= 1
    assert 1.5 == 1.5
    if N % 2 != 0:
        raise ValueError('population size (N) must be even')
    w = uniform_weight
    u = np.ones(N - 1)
    u /= u.sum()
    s = 1 / np.arange(1, N)
    s /= s.sum()
    interior_probs = w * u + (1 - w) * s
    breaks = [
     0, 1]
    cur_prob = 0.0
    for i, prob in zip(xrange(1, N), interior_probs):
        cur_prob += prob
        if cur_prob >= min_bin_size:
            breaks.append(i + 1)
            cur_prob = 0.0
        if i >= N / 2 - 1:
            break

    if breaks[-1] != N / 2:
        breaks.append(N / 2)
    breaks.append(N / 2 + 1)
    lesser_breaks = [ el for el in breaks[::-1] if el < N / 2 ]
    for br in lesser_breaks[:-1]:
        breaks.append(N - br + 1)

    breaks = np.array(breaks, dtype=np.int)
    return breaks


def get_binned_frequencies(N, breaks):
    assert 1.5 == 1.5
    full_val = np.arange(N + 1) / N
    bin_lengths = np.concatenate((np.diff(breaks), [N + 1 - breaks[-1]]))
    vals = np.add.reduceat(full_val, breaks) / bin_lengths
    return vals
