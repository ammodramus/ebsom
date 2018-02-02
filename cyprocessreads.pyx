from pysam.libcalignmentfile cimport AlignmentFile
from pysam.libcalignedsegment cimport AlignedSegment
import util as ut
cimport numpy as np

            #pr.add_observations(read, mapq, min_bq, context_len, rm,
            #        bam_fn, ref, consensus, is_can, nc_observations, c_observations)
cpdef void add_observations(
        AlignedSegment read,
        int mapq,
        int min_bq,
        int context_len,
        rm,
        str bam_fn,
        str ref,
        str consensus,
        is_candidate,
        nc_observations,
        c_observations):
    cdef:
        str seq, context, obsbase, consbase
        int readlen, readnum, qpos, q, dend, refpos
        unsigned char[:] qualities
        bint reverse

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
            context = ut.rev_comp(seq[qpos+1:qpos+3])
            if 'N' in context:
                continue
            obsbase = ut.rev_comp(seq[qpos])
            consbase = ut.rev_comp(consensus[refpos])
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
        row = rm[consbase].get_covariate_row(consbase, q, mapq,
                context, dend, refpos, bam_fn, reverse)
        if not is_candidate[refpos]:
            nc_observations.add(row, consbase, readnum, obsbase)
        else:
            c_observations.add(row, bam_fn, ref, refpos, reverse,
                    consbase, readnum, obsbase)

def add_bam_observations(
        AlignmentFile bam,
        str ref,
        int reflen,
        int min_bq,
        int min_mq,
        int context_len,
        rm,
        str bam_fn,
        str consensus,
        is_candidate,
        nc_observations,
        c_observations):

    cdef:
        AlignedSegment read
        int mapq, i

    i = 0
    for read in bam.fetch(contig = ref, start = 0, end = reflen):
        mapq = read.mapping_quality
        if mapq < min_mq:
            continue
        add_observations(read, mapq, min_bq, context_len, rm,
                bam_fn, ref, consensus, is_candidate, nc_observations,
                c_observations)
        print('obs added', i)
        i += 1
        if i >= 1000:
            break
