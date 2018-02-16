import likelihood as lik


@jit calc_g_p(betas, X_i, Y_i, logprobs):
    '''
    betas: parameters for this regression, fr. rowlen x 3 (or 4?)
    X_i: covariate vector for this read
    Y_i: outcome for this observation
    logprobs: log-probabilities for the different outcomes
    '''
    pass


@jit
def calc_loga_p(
        grad,
        params,
        blims,
        lp_maj_f1,
        lp_maj_f2,
        lp_min_f1,
        lp_min_f2,
        lp_maj_r1,
        lp_maj_r2,
        lp_min_r1,
        lp_min_r2,
        lo_f1,
        lo_f2,
        lo_r1,
        lo_r2,
        cm, logpf, lf, l1mf):
    '''
    needed for loga_p:
    params
    cm  (for X values)
    logobs
    logprobs
    '''
    ll = 0.0
    nj = logpf.shape[0]  # number of frequencies
    Ma = -1e100  # very small number... DBL_MAX is ~1e308
    s = np.tile(logpf.copy(), params.shape[0]).reshape((-1,logpf.shape[0]))
    for i in range(nj):
        # for each f, have to calculate logb_p for a whole vector, and sum these using logsumexp
        tlf = lf[i]
        tl1mf = l1mf[i]
        s[i] += calc_logb_p(s, params, blims, lp_maj_f1, lp_min_f1, lo_f1, tlf, tl1mf)

        tot_fll = 0.0
        # calculate the fll's for different fr/12's
        tot_fll += calc_loc_ll_cond_f_and_fr(
                lp_maj_f1,
                lp_min_f1,
                lo_f1,
                tlf,
                tl1mf)
        tot_fll += calc_loc_ll_cond_f_and_fr(
                lp_maj_f2,
                lp_min_f2,
                lo_f2,
                tlf,
                tl1mf)
        tot_fll += calc_loc_ll_cond_f_and_fr(
                lp_maj_r1,
                lp_min_r1,
                lo_r1,
                tlf,
                tl1mf)
        tot_fll += calc_loc_ll_cond_f_and_fr(
                lp_maj_r2,
                lp_min_r2,
                lo_r2,
                tlf,
                tl1mf)
        a[i] += tot_fll
        if a[i] > Ma:
            Ma = a[i]
    locll = 0   # logsumexp routine here
    for el in a:
        locll += math.exp(el-Ma)
    locll = math.log(locll) + Ma
    return locll


def calc_loga_p_with_mm(grad, params, logprobs, cm, locobs, major, minor, blims, rowlen,
        freqs, breaks, logpf, lf, l1mf, regs):
    rmajor, rminor = ut.rev_comp(major), ut.rev_comp(minor)
    lp_maj_f1 = logprobs[(major,1)]
    lp_maj_f2 = logprobs[(major,2)]
    lp_maj_r1 = logprobs[(rmajor,1)]
    lp_maj_r2 = logprobs[(rmajor,2)]
    lo_f1 = locobs[0][0]
    lo_f2 = locobs[0][1]
    lo_r1 = locobs[1][0]
    lo_r2 = locobs[1][1]

    if minor != 'N':
        lp_min_f1 = logprobs[(minor,1)]
        lp_min_f2 = logprobs[(minor,2)]
        lp_min_r1 = logprobs[(rminor,1)]
        lp_min_r2 = logprobs[(rminor,2)]
        calc_loga_p(
                grad, params,, blims,
                lp_maj
                lp_maj_f1,
                lp_maj_f2,
                lp_min_f1,
                lp_min_f2,
                lp_maj_r1,
                lp_maj_r2,
                lp_min_r1,
                lp_min_r2,
                lo_f1,
                lo_f2,
                lo_r1,
                lo_r2,
                cm, logpf, lf, l1mf)


def gradient(params, ref, bam, position, cm, lo, mm, blims,
        rowlen, freqs, breaks, lf, l1mf, regs):

    betas = params[:-2]
    ab, ppoly = params[-2:]
    N = 1000
    logpf = np.log(afd.get_stationary_distribution_double_beta(
        freqs, breaks, N, ab, ppoly))

    ll = single_locus_log_likelihood(params, ref, bam, position, cm, lo, mm, blims,
            rowlen, freqs, breaks, lf, l1mf, regs)
    logprobs = {}
    X = cm
    for reg in regs:
        low, high = blims[reg]
        b = betas[low:high].reshape((rowlen,-1))
        Xb = np.column_stack((np.dot(X,b), np.zeros(X.shape[0])))
        Xb -= logsumexp(Xb, axis = 1)[:,None]
        logprobs[reg] = Xb

    grad = np.zeros_like(params)

    locobs = lo[ref][bam][position]
    major, minor = mm[ref][bam][position]

    loga_p = calc_loga_p_with_mm(grad, params, logprobs, cm, locobs, major, minor, blims, rowlen,
            freqs, logpf, breaks, lf, l1mf, regs)

    return np.exp(loga_p - ll)
