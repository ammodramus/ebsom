import os.path as osp
import pysam
from rowmaker import CovariateRowMaker

def positive_int(val):
    v = int(val)
    if v <= 0:
        raise ValueError('invalid positive integer {}'.format(val))
    return v


def nonneg_int(val):
    v = int(val)
    if v < 0:
        raise ValueError('invalid positive integer {}'.format(val))
    return v

def probability(val):
    v = float(val)
    if v < 0 or v > 1:
        raise ValueError('invalid probability {}'.format(val))
    return v

BASES = 'ACGT'
RCBASES = 'TGCA'
def rev_comp(seq):
    try:
        return ''.join([RCBASES[BASES.index(b)] for b in seq])
    except ValueError:
        raise ValueError('invalid base')
    raise ValueError('problem in revcomp')
        
def get_counts(aln_file, *args, **kwargs):
    counts = aln_file.count_coverage(*args, **kwargs)
    counts = np.array([np.array(arr) for arr in counts])
    counts = np.transpose(counts)
    return counts

def get_consensus(freqs, min_cov = 1):
    consensus = np.array(list('ACGT'))[freqs.argmax(1)]
    consensus[freqs.sum(1) < min_cov] = 'N'
    return consensus

def get_bams(bams_fn):
    '''
    bams_fn       file with a list of bams

    Returns
        tuple of bam_fns (list) and bams (dict)
    '''
    bam_fns = []
    bams = {}
    with open(bams_fn) as fin:
        for line in bams_fn:
            bam_fn = line.strip()
            if not osp.isfile(bam_fn):
                raise ValueError('could not find bam file {}'.format(bam_fn))
            bam_fns.append(bam_fn)
            bam = pysam.AlignmentFile(bam_fn)
            bams[bam_fn] = bam
    return bam_fns, bams

def get_ref_names(refs_input, bams):
    if osp.isfile(refs_input):
        ref_names = []
        with open(refs_input) as fin:
            for line in fin:
                ref_names.append(line.strip())
    else:
        ref_names = [el.strip() for el in refs_input.split(',')]
    for bam_fn, bam in bams.iteritems():
        for ref in ref_names:
            if ref not in bam.references:
                err = 'could not find reference {} in bam file {}'.format(
                        ref, bam_fn)
                raise ValueError(err)
    return ref_names


# these will be memory hogs
def get_all_counts(bams, refs, min_bq):
    '''
    bams      dict of bam_fn:AlignmentFiles
    refs      list of reference names
    min_bq    minimum base quality

    Returns
        a dictionary of counts, where counts[bam_fn][ref] gives the array of
        counts for that bam and reference sequence
    '''
    counts = {}
    for bam_fn, bam in bams.iteritems():
        counts[bam_fn] = {}
        bamref = bam.header['SQ']
        for ref in refs:
            reflen_found = False
            for d in bamref:
                if d['SN'] == ref:
                    reflen = d['LN']
                    reflen_found = True
                    break
            if not reflen_found:
                raise ValueError('length of ref {} not found in {}'.format(
                    ref, bam_fn))
            counts = get_counts(bam, contig=ref, start = 0, end = reflen,
                    quality_threshold = min_bq)
            counts[bam_fn][ref] = counts
    return counts

def get_freqs(counts):
    freqs = {}
    for bam_fn, bam_counts in counts.iteritems():
        freqs[bam_fn] = {}
        for ref, c in bam_counts.iteritems():
            # using jit or cython, this can be done with 1/2 the memory
            f = c / np.maximum(c,1)[:,None]
            freqs[bam_fn][ref] = f
    return freqs

def determine_candidates(freqs, min_candidate_freq):
    candidates = {}
    for bam_fn, bam_freqs in freqs.iteritems():
        candidates[bam_fn] = {}
        for ref, f in bam_freqs.iteritems():
            is_candidate = (f > min_candidate_freq).sum(1) > 1
            candidates[bam_fn][ref] = is_candidate
    return candidates

def get_all_consensuses(counts, min_coverage):
    all_con = {}
    for bam_fn, bam_counts in counts.iteritems():
        all_con[ref] = {}
        for ref, c in bam_counts.iteritems():
            consensus = get_consensus(c, min_cov=min_coverage)

def get_row_makers(bam_fns, refs, context_len, dend_roundby, consensuses):
    '''
    bam_fns             list of bam filenames
    refs                list of reference sequence names
    context_len         number of preceding bases to include as covariates
    dend_roundby        how much to round distance from beginning by
    consensuses         dict of dicts, consensuses[bam_fn][ref] gives the
                        consensus sequence for ref sequence ref in bam file 
                        bam_fn
    
    Returns tuple (d, rowlen)
        d is dict of dict of dict of CovariateRowMakers:
        rm[bam_fn][ref][base] gives the CovariateRowMaker for this bam, ref,
        and base

        rowlen is the length of a row in the matrix
    '''
    rm = {}
    for bam_fn in bam_fns:
        rm[bam_fn] = {}
        for ref in refs:
            rm[bam_fn][ref] = {}
            cons = consensuses[bam_fn][ref]
            other_cons = [
                    consensuses[fn][ref] for fn in bam_fns if fn != bam_fn]
            for base in 'ACGT':
                rm[bam_fn][ref][base] = CovariateRowMaker(
                    base,
                    context_len,
                    dend_roundby,
                    cons,
                    other_cons)
    
    rowlen = rm[bam_fn][ref][base].rowlen
    return rm, rowlen
