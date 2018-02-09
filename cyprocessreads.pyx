from __future__ import print_function
from pysam.libcalignmentfile cimport AlignmentFile
from pysam.libcalignedsegment cimport AlignedSegment
import util as ut
cimport cyutil as cut
cimport numpy as np
import sys

from cylocobs cimport LocObs
from cyregcov cimport RegCov

cdef const char *BASES = 'ACGT'
cdef inline int get_base_idx(bytes obsbase, bytes true):
    cdef:
        int i, found
        char o, t, c
    o = obsbase[0]
    t = true[0]
    if o == t:
        return 3   # last column

    found = 0
    for i in range(4):
        c = BASES[i]
        if c == o:
            return i-found
        if c == t:
            found = 1
    return -1

# obs true   seq   found  i  ret
# A   G      ACTG  0      0  0
# T   C      AGTC  1      3  2


#cpdef void add_observations(
def add_observations(
        AlignedSegment read,
        int mapq,
        int min_bq,
        int context_len,
        rm,
        bytes bam_fn,
        bytes ref,
        bytes consensus,
        covariate_matrices,
        locus_observations,
        major_alleles):
    cdef:
        bytes seq, context, obsbase, consbase
        int readlen, readnum, qpos, q, dend, refpos, cov_idx, base_idx
        unsigned char[:] qualities
        bint reverse
        RegCov cov
        LocObs loc
        

    seq = read.seq
    readlen = read.alen  # aligned length
    qualities = read.query_qualities
    reverse = read.is_reverse
    readnum = read.is_read2 + 1
    for qpos, refpos in read.get_aligned_pairs(True):
        if qpos is None or refpos is None:
            continue
        q = qualities[qpos]
        if q < min_bq:
            continue
        if reverse:
            if qpos >= readlen-context_len:
                continue
            context = cut.rev_comp(seq[qpos+1:qpos+3])
            if 'N' in context:
                continue
            obsbase = cut.rev_comp(seq[qpos])
            consbase = cut.rev_comp(consensus[refpos])
            dend = readlen - qpos
        else:
            if qpos < context_len:
                continue
            context = seq[qpos-context_len:qpos]
            if 'N' in context:
                continue
            obsbase = seq[qpos]
            consbase = consensus[refpos]
            dend = qpos
        if obsbase == 'N' or consbase == 'N':
            continue
        if consbase not in rm:
            import sys
            print(consbase, file=sys.stderr)
            raise ValueError('not here')
        row = rm[consbase].get_covariate_row(consbase, q, mapq,
                context, dend, refpos, bam_fn, reverse)

        major = major_alleles[refpos][0]
        if major == 'N':
            continue
        if reverse:
            major = cut.rev_comp(major)
        cov = covariate_matrices[(major, readnum)]
        cov_idx = cov.set_default(row)
        base_idx = get_base_idx(obsbase, major)
        if base_idx < 0 or base_idx > 3:
            raise ValueError('invalid base')
        mmidx = 0
        revidx = int(reverse)
        ridx = readnum-1
        loc = locus_observations[refpos][mmidx][revidx][ridx]
        loc.add_obs(cov_idx, base_idx)

        minor = major_alleles[refpos][1]
        if minor == 'N':
            continue
        if reverse:
            minor = cut.rev_comp(minor)
        cov = covariate_matrices[(minor, readnum)]
        cov_idx = cov.set_default(row)
        base_idx = get_base_idx(obsbase, minor)
        if base_idx < 0 or base_idx > 3:
            raise ValueError('invalid base')
        mmidx = 1
        revidx = int(reverse)
        ridx = readnum-1
        loc = locus_observations[refpos][mmidx][revidx][ridx]
        loc.add_obs(cov_idx, base_idx)

def add_bam_observations(
        AlignmentFile bam,
        bytes ref,
        int reflen,
        int min_bq,
        int min_mq,
        int context_len,
        rm,
        bytes bam_fn,
        bytes consensus,
        covariate_matrices,
        locus_observations,
        major_alleles,
        int update_interval = 1000,
        max_num_reads = 1000):

    cdef:
        AlignedSegment read
        int mapq, i

    i = 0
    for read in bam.fetch(contig = ref, start = 0, end = reflen):
        mapq = read.mapping_quality
        if mapq < min_mq:
            continue
        add_observations(read, mapq, min_bq, context_len, rm,
                bam_fn, ref, consensus, covariate_matrices,
                locus_observations, major_alleles)
        if i % update_interval == 0:
            print('{}, {}: processed {} of {} reads'.format(
                bam_fn, ref, i, bam.mapped), file = sys.stderr)
        i += 1
        if i == max_num_reads:
            break
