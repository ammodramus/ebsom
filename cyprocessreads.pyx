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

from collections import defaultdict

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
        batch_locobs,
        major_alleles,
        int min_refpos,
        int max_refpos):
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
    
    cdef bytes bseq = read.seq
    cdef char *cseq = bseq
    cdef char obsbase, consbase

    cdef char *cconsensus = consensus
    readlen = read.alen  # aligned length
    reverse = read.is_reverse
    readnum = read.is_read2 + 1

    cdef np.ndarray[np.uint32_t,ndim=2] al_pairs_np = np.array(read.get_aligned_pairs(matches_only = True), dtype = np.uint32)
    cdef uint32_t [:,:] al_pairs = al_pairs_np

    cdef uint32_t i, j
    for i in range(al_pairs_np.shape[0]):
        qpos = al_pairs[i,0]
        refpos = al_pairs[i,1]
        if refpos < min_refpos or refpos > max_refpos:
            continue
        q = qualities[qpos]
        if q < min_bq:
            continue
        if reverse:
            if qpos >= readlen-context_len:
                continue
            #context = cut.comp(bseq[qpos+1:qpos+1+context_len])
            context = cut.comp(bseq[qpos+context_len:qpos:-1])
            #print('rev context:', context, 'previously', cut.comp(bseq[qpos+1:qpos+1+context_len]))
            if 'N' in context:
                continue
            obsbase = cut.comp1(cseq[qpos])
            consbase = cut.comp1(cconsensus[refpos])
            dend = readlen - qpos
        else:
            if qpos < context_len:
                continue
            context = bseq[qpos-context_len:qpos]
            if 'N' in context:
                continue
            obsbase = cseq[qpos]
            consbase = cconsensus[refpos]
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
        loc = batch_locobs[refpos][reverse][readnum-1]
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
        h5lo_bam,
        major_alleles,
        int update_interval = 1000,
        read_batch_size = 500,
        max_num_reads = -1):

    cdef:
        AlignedSegment read
        int mapq, i

    
    # locobs are built read by read, not position by position.
    # add reads to h5lo in 500-bp windows, using fetch.
    # keep a dict of already-processed reads, skip if already seen.


    i = 0
    
    #reads_seen = set([])
    for start_bp in range(0, reflen, read_batch_size):
        end_bp = start_bp + read_batch_size
        batch_locobs = defaultdict(lambda: ((LocObs(), LocObs()), (LocObs(), LocObs())))

        for read in bam.fetch(contig = ref, start = start_bp, end = end_bp):
            if i % update_interval == 0:
                print('{}, {}: processed {} of {} reads'.format(
                    bam_fn, ref, i, bam.mapped), file = sys.stderr)
            i += 1
            if i == max_num_reads:
                break
            mapq = read.mapping_quality
            if mapq < min_mq:
                continue
            #qn = read.query_name 
            #if qn in reads_seen:
            #    continue
            #reads_seen.add(qn)  # not adding reads failing mapq filter, presumably faster
            #                    # to get mapq than hash name

            add_observations(read, mapq, min_bq, context_len, rm,
                    bam_fn, ref, consensus, covariate_matrix,
                    batch_locobs, major_alleles, start_bp, end_bp-1)
        
        add_batch_locobs(batch_locobs, h5lo_bam)

        if i == max_num_reads:
            break


def add_batch_locobs(batch_locobs, h5lo_bam):
    for loc_idx, loc_obs in batch_locobs.iteritems():
        if str(loc_idx) in h5lo_bam.keys():
            print('already there:', str(loc_idx))
        h5lo_loc = h5lo_bam.create_group(str(loc_idx))
        if str(loc_idx) in h5lo_loc:
            raise ValueError('already in hdf5 file!')
        f1dat = loc_obs[0][0].counts().astype(np.uint32)
        h5lo_loc.create_dataset('f1', dtype = np.uint32, data = f1dat)
        f2dat = loc_obs[0][1].counts().astype(np.uint32)
        h5lo_loc.create_dataset('f2', dtype = np.uint32, data = f2dat)
        r1dat = loc_obs[1][0].counts().astype(np.uint32)
        h5lo_loc.create_dataset('r1', dtype = np.uint32, data = r1dat)
        r2dat = loc_obs[1][1].counts().astype(np.uint32)
        h5lo_loc.create_dataset('r2', dtype = np.uint32, data = r2dat)
