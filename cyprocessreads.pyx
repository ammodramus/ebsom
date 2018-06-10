## cython: profile=True
## cython: linetrace=True
## cython: binding=True
## distutils: define_macros=CYTHON_TRACE_NOGIL=1

# cython: wraparound=False
# cython: boundscheck=False

from __future__ import print_function
from pysam.libcalignmentfile cimport AlignmentFile
from pysam.libcalignedsegment cimport AlignedSegment
import util as ut
cimport cyutil as cut
cimport numpy as np
import sys

import numpy as np

from pysam.libchtslib cimport *
from pysam.libcsamfile cimport *

from cylocobs cimport LocObs
from cyregcov cimport RegCov
from cyrowmaker cimport CyCovariateRowMaker

cdef inline int get_base_idx(char obsbase):
    if obsbase == 'A':
        return 0
    if obsbase == 'C':
        return 1
    if obsbase == 'G':
        return 2
    if obsbase == 'T':
        return 3
    return -1


'''
these two functions from libcalignedsegment.pyx. (not in pysam's
libcalignedsegment.pxd, unfortunately.) check license.
'''

cdef inline int32_t getQueryStart(bam1_t *src) except -1:
    cdef uint32_t * cigar_p
    cdef uint32_t start_offset = 0
    cdef uint32_t k, op

    cigar_p = pysam_bam_get_cigar(src);
    for k from 0 <= k < pysam_get_n_cigar(src):
        op = cigar_p[k] & BAM_CIGAR_MASK
        if op == BAM_CHARD_CLIP:
            if start_offset != 0 and start_offset != src.core.l_qseq:
                raise ValueError('Invalid clipping in CIGAR string')
        elif op == BAM_CSOFT_CLIP:
            start_offset += cigar_p[k] >> BAM_CIGAR_SHIFT
        else:
            break

    return start_offset


cdef inline int32_t getQueryEnd(bam1_t *src) except -1:
    cdef uint32_t * cigar_p = pysam_bam_get_cigar(src)
    cdef uint32_t end_offset = src.core.l_qseq
    cdef uint32_t k, op

    # if there is no sequence, compute length from cigar string
    if end_offset == 0:
        for k from 0 <= k < pysam_get_n_cigar(src):
            op = cigar_p[k] & BAM_CIGAR_MASK
            if op == BAM_CMATCH or \
               op == BAM_CINS or \
               op == BAM_CEQUAL or \
               op == BAM_CDIFF or \
              (op == BAM_CSOFT_CLIP and end_offset == 0):
                end_offset += cigar_p[k] >> BAM_CIGAR_SHIFT
    else:
        # walk backwards in cigar string
        for k from pysam_get_n_cigar(src) > k >= 1:
            op = cigar_p[k] & BAM_CIGAR_MASK
            if op == BAM_CHARD_CLIP:
                if end_offset != src.core.l_qseq:
                    raise ValueError('Invalid clipping in CIGAR string')
            elif op == BAM_CSOFT_CLIP:
                end_offset -= cigar_p[k] >> BAM_CIGAR_SHIFT
            else:
                break

    return end_offset

def add_observations(
        AlignedSegment read,
        int mapq,
        int min_bq,
        int context_len,
        CyCovariateRowMaker rm,
        bytes bam_fn,
        bytes ref,
        bytes consensus,
        covariate_matrix,
        locus_observations,
        major_alleles):
    cdef:
        #bytes seq, context, obsbase, consbase
        #bytes seq, context
        bytes context
        int readlen, readnum, qpos, q, dend, refpos, cov_idx, base_idx
        bint reverse
        RegCov cov
        LocObs loc
        
   
    #cdef unsigned char[:] qualities = read.query_qualities
    cdef uint8_t *qualities = pysam_bam_get_qual(read._delegate)
    
    cdef char *seq = read.seq
    cdef char obsbase, consbase
    readlen = read.alen  # aligned length
    reverse = read.is_reverse
    readnum = read.is_read2 + 1

    cdef np.ndarray[np.uint32_t,ndim=2] al_pairs_np = np.array(read.get_aligned_pairs(matches_only = True), dtype = np.uint32)
    cdef uint32_t [:,:] al_pairs = al_pairs_np

    cdef uint32_t i, j
    for i in range(al_pairs_np.shape[0]):
        qpos = al_pairs[i,0]
        refpos = al_pairs[i,1]
        q = qualities[qpos]
        if q < min_bq:
            continue
        if reverse:
            if qpos >= readlen-context_len:
                continue
            context = cut.comp(seq[qpos+1:qpos+1+context_len])
            if 'N' in context:
                continue
            obsbase = cut.comp(seq[qpos])
            consbase = cut.comp(consensus[refpos])
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
        row = rm.get_covariate_row(q, mapq, context, dend, refpos, bam_fn,
                reverse)

        major = major_alleles[refpos][0]
        if major == 'N':
            continue
        if reverse:
            major = cut.comp(major)
        cov_idx = covariate_matrix.set_default(row)
        base_idx = get_base_idx(obsbase)
        loc = locus_observations[refpos][reverse][readnum-1]
        loc.add_obs(cov_idx, base_idx)

def add_bam_observations(
        AlignmentFile bam,
        bytes ref,
        int reflen,
        int min_bq,
        int min_mq,
        int context_len,
        CyCovariateRowMaker rm,
        bytes bam_fn,
        bytes consensus,
        covariate_matrix,
        locus_observations,
        major_alleles,
        int update_interval = 1000,
        max_num_reads = -1):

    cdef:
        AlignedSegment read
        int mapq, i

    i = 0
    for read in bam.fetch(contig = ref, start = 0, end = reflen):
        mapq = read.mapping_quality
        if mapq < min_mq:
            continue
        add_observations(read, mapq, min_bq, context_len, rm,
                bam_fn, ref, consensus, covariate_matrix,
                locus_observations, major_alleles)
        if i % update_interval == 0:
            print('{}, {}: processed {} of {} reads'.format(
                bam_fn, ref, i, bam.mapped), file = sys.stderr)
        i += 1
        if i == max_num_reads:
            break
