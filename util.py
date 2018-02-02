import os.path as osp
import pysam
from rowmaker import CovariateRowMaker
import numpy as np
from numba import jit

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

REVCOMPBASES='ACGTN'
REVCOMPRCBASES='TGCAN'
def rev_comp(seq):
    try:
        return ''.join([REVCOMPRCBASES[REVCOMPBASES.index(b)] for b in seq])
    except ValueError:
        raise ValueError('invalid base in {}'.format(seq))
    raise ValueError('problem in revcomp')
        
def get_counts(aln_file, *args, **kwargs):
    counts = aln_file.count_coverage(*args, **kwargs)
    counts = np.array([np.array(arr) for arr in counts])
    counts = np.transpose(counts)
    return counts

def get_consensus(counts, min_cov = 1):
    consensus = np.array(list('ACGT'))[counts.argmax(1)]
    consensus[counts.sum(1) < min_cov] = 'N'
    consensus = ''.join(list(consensus))
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
        for line in fin:
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
        a dictionary of counts, where counts[ref][bam_fn] gives the array of
        counts for that reference sequence and bam file
    '''

    counts = {}
    for ref in refs:
        counts[ref] = {}
        for bam_fn, bam in bams.iteritems():
            bamref = bam.header['SQ']
            reflen_found = False
            for d in bamref:
                if d['SN'] == ref:
                    reflen = d['LN']
                    reflen_found = True
                    break
            if not reflen_found:
                raise ValueError('length of ref {} not found in {}'.format(
                    ref, bam_fn))
            c = get_counts(bam, contig=ref, start = 0, end = reflen,
                    quality_threshold = min_bq)
            counts[ref][bam_fn] = c
    return counts

def get_freqs(counts):
    freqs = {}
    for ref, ref_counts in counts.iteritems():
        freqs[ref] = {}
        for bam_fn, c in ref_counts.iteritems():
            # using jit or cython, this can be done with 1/2 the memory
            f = c / np.maximum(c.sum(1),1)[:,None]
            freqs[ref][bam_fn] = f
    return freqs

def determine_candidates(freqs, min_candidate_freq):
    candidates = {}
    for ref, ref_freqs in freqs.iteritems():
        candidates[ref] = {}
        for bam_fn, f in ref_freqs.iteritems():
            is_candidate = (f > min_candidate_freq).sum(1) > 1
            candidates[ref][bam_fn] = is_candidate
    return candidates

def get_all_consensuses(counts, min_coverage):
    all_con = {}
    for ref, ref_counts in counts.iteritems():
        all_con[ref] = {}
        for bam_fn, c in ref_counts.iteritems():
            consensus = get_consensus(c, min_cov=min_coverage)
            all_con[ref][bam_fn] = consensus
    return all_con

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
        rm[ref][bam_fn][base] gives the CovariateRowMaker for this bam, ref,
        and base

        rowlen is the length of a row in the matrix
    '''
    rm = {}
    for ref in refs:
        rm[ref] = {}
        for bam_fn in bam_fns:
            rm[ref][bam_fn] = {}
            cons = consensuses[ref][bam_fn]
            other_cons = [
                    consensuses[ref][fn] for fn in bam_fns if fn != bam_fn]
            for base in 'ACGT':
                rm[ref][bam_fn][base] = CovariateRowMaker(
                    base,
                    context_len,
                    dend_roundby,
                    cons,
                    other_cons)
    
    rowlen = rm[ref][bam_fn][base].rowlen
    return rm, rowlen
