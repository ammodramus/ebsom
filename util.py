from __future__ import division, print_function
import os.path as osp
import pysam
#from rowmaker import CovariateRowMaker
from cyrowmaker import CyCovariateRowMaker
from rowmaker import CovariateRowMaker
import numpy as np
from numba import jit
from cylocobs import LocObs
from cyregcov import RegCov
import os.path

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
        tuple of basenames of bam fns (list), a prefix, and bams (dict, key is
        basename)
    '''
    bam_fns = []
    bams = {}
    prefix = None
    with open(bams_fn) as fin:
        for line in fin:
            bam_fp = line.strip()
            head, bam_fn = os.path.split(bam_fp)
            if prefix is not None:
                if head != prefix:
                    raise ValueError('all bam files must be in the same directory')
            else:
                prefix = head
            if not osp.isfile(bam_fp):
                raise ValueError('could not find bam file {}'.format(bam_fn))
            if bam_fn in bam_fns:
                raise ValueError('all BAM filenames must be unique')
            bam_fns.append(bam_fn)
            bam = pysam.AlignmentFile(bam_fp)
            bams[bam_fn] = bam
    return prefix, bam_fns, bams

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

def get_row_makers(bam_fns, refs, context_len, dend_roundby, consensuses,
        use_mapq):
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
    rowlen = None
    for ref in refs:
        rm[ref] = {}
        for bam_fn in bam_fns:
            cons = consensuses[ref][bam_fn]
            other_cons = [
                    consensuses[ref][fn] for fn in bam_fns if fn != bam_fn]
            thisrm = CyCovariateRowMaker(
                context_len,
                dend_roundby,
                cons,
                other_cons,
                bam_fns,
                use_mq = use_mapq)
            rm[ref][bam_fn] = thisrm
            l = thisrm.rowlen
            if rowlen is None:
                rowlen = l
            else:
                assert rowlen == l, 'error: multiple row lengths'
    rowlen = rm[ref][bam_fn].rowlen
    return rm, rowlen

def get_all_majorminor(all_counts):
    all_majorminor = {}
    allbases = np.array(list('ACGT'))
    for ref, refcounts in all_counts.iteritems():
        all_majorminor[ref] = {}
        for bamname, bamcounts in refcounts.iteritems():
            all_majorminor[ref][bamname] = []
            thismm = all_majorminor[ref][bamname]
            # bamcounts is a 2-d np.ndarray, shape (seqlen, 4)
            # (yes, it does provide zero-counts for position at which
            #  no reads aligned)
            for i, order in enumerate(bamcounts.argsort(1)):
                order = order[::-1]
                counts = bamcounts[i,order]
                bases = allbases[order]
                major = bytes(bases[0]) if counts[0] > 0 else b'N'
                # note that minor is arbitrarily decided if there's a tie
                minor = bytes(bases[1]) if counts[1] > 0 else b'N'
                thismm.append((major, minor))
    return all_majorminor

def get_covariate_matrices(rowlen):
    covmat = RegCov(rowlen)
    return covmat


def get_locus_observations(all_majorminor):
    locobs = {}
    for ref, refmm in all_majorminor.iteritems():
        locobs[ref] = {}
        for bam, bammm in refmm.iteritems():
            locobs[ref][bam] = []
            for i, _ in enumerate(bammm):
                # indexed [forward(0)/reverse(1)][read1[0]/read2[1]]
                locobs[ref][bam].append(((LocObs(),LocObs()),(LocObs(),LocObs())))
    return locobs

def collect_covariate_matrices(cov):
    ret = cov.covariate_matrix()
    return ret

def collect_indiv_loc_obs(obs):
    c = lambda x: x.counts()
    m = obs
    col = ( (c(m[0][0]), c(m[0][1])), (c(m[1][0]), c(m[1][1])) )
    return col


def collect_loc_obs(locobs):
    ret = {}
    for ref, refobs in locobs.iteritems():
        ret[ref] = {}
        for bam, bamobs in refobs.iteritems():
            ret[ref][bam] = []
            for lobs in bamobs:
                ret[ref][bam].append(collect_indiv_loc_obs(lobs))
    return ret

def sort_lo(lo):
    lo_sorted = {}
    for chrom, lochrom in lo.iteritems():
        lo_sorted[chrom] = {}
        for bam_fn, lobam in lochrom.iteritems():
            lo_sorted[chrom][bam_fn] = []
            for locidx, loclo in enumerate(lobam):
                thislo = []
                for fr in [0,1]:
                    p = []
                    for r in [0,1]:
                        a = loclo[fr][r]
                        p.append(a[np.argsort(a[:,0])].astype(np.int32).copy())
                    p = tuple(p)
                    thislo.append(p)
                thislo = tuple(thislo)
                lo_sorted[chrom][bam_fn].append(thislo)
    return lo_sorted

def normalize_covariates(cm):
    retcm = cm.copy()
    min_maxes = []
    # starting with second column because first is constant!
    for j in range(1,cm.shape[1]):
        m = retcm[:,j].min()
        M = retcm[:,j].max()
        if m != M:
            retcm[:,j] = (retcm[:,j]-m)/(M-m)
        else:
            retcm[:,j] = 0
        min_maxes.append((m,M))
    return retcm, min_maxes
