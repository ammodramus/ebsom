## cython: profile=True
## cython: linetrace=True
## cython: binding=True
## distutils: define_macros=CYTHON_TRACE_NOGIL=1

# cython: wraparound=False
# cython: boundscheck=True

from __future__ import print_function
cimport cyutil as cut
from libc.stdint cimport *

from pysam.libcalignmentfile cimport AlignmentFile, IteratorColumnRegion
from pysam.libcalignedsegment cimport AlignedSegment, PileupColumn, PileupRead
from pysam.libchtslib cimport *
from pysam.libcsamfile cimport *

from cyregcov cimport RegCov

cimport numpy as np
import numpy as np

np.import_array()

cdef char * my_seq_nt16_str = "=ACMGRSVTWYHKDBN";
cdef char * my_seq_nt16_str_rev = "=TGMCRSVAWYHKDBN";  # note: assuming that non ACGT bases aren't used

def complement(bytes bases):
    cBASES = 'TGCA'
    cbases = []
    cdef char *ret = <char *>malloc(sizeof(char)*(len(bases)+1))
    cdef int idx
    for base in bases:
        idx = get_base_idx(<char>(base[0]))
        cbases.append(cBASES[idx])
    return ''.join(cbases)


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


def get_column_data(
        AlignmentFile bam,
        bytes ref,
        int reflen,
        int base_position,
        int min_bq,
        int min_mq,
        int context_len,
        bytes consensus,
        int round_distance_by):

    cdef:
        AlignedSegment aln
        IteratorColumnRegion itcol
        PileupColumn pcol
        PileupRead pread
        list pileups
        uint32_t isread2, isreverse, qlen, k, dend
        uint16_t qpos
        uint8_t *p
        uint8_t bq, mq
        char qbase
        size_t idx
        
    nobs = 0

    covrow = np.zeros(4, dtype = np.float32)
    k = base_position


    counts_forward = RegCov(4)
    counts_reverse = RegCov(4)
    itcol = bam.pileup(contig = ref, start = base_position, stop = base_position+1, truncate = True, min_base_quality = min_bq, min_mapping_quality = min_mq, ignore_orphans = False)
    pcol = itcol.next()
    pileups = pcol.pileups
    for pread in pileups:
        if pread._is_del or pread._is_refskip:
            continue
        aln = pread._alignment
        qpos = pread._qpos
        mq = aln.mapping_quality
        isread2 = (aln.flag & BAM_FREAD2) != 0
        qlen = aln._delegate.core.l_qseq
        isreverse = (aln.flag & BAM_FREVERSE) != 0
        k = qpos
        p = pysam_bam_get_seq(aln._delegate)
        idx = p[k/2] >> 4 * (1-k%2) & 0xf
        if isreverse:
            qbase = my_seq_nt16_str_rev[idx]
        else:
            qbase = my_seq_nt16_str[idx]
        bq = pysam_bam_get_qual(aln._delegate)[k]
        dend = qpos if isreverse else qlen-qpos-1
        dend = (dend // round_distance_by) * round_distance_by
        covrow[0] = <float>bq
        covrow[1] = <float>mq
        covrow[2] = <float>dend
        covrow[3] = <float>isread2
        if isreverse:
            counts_reverse.set_default(covrow, get_base_idx(qbase))
        else:
            counts_forward.set_default(covrow, get_base_idx(qbase))
    return counts_forward, counts_reverse
