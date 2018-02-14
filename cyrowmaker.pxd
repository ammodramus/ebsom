cdef class CyCovariateRowMaker:
    cdef:
        int contextlen, ncontexts, dend_roundby, user_ncols, bam_idx, num_bams
        int context_index
        bytes thiscons
        list othercons
        dict userdefined_rows, bam_idxs
        public list userdefined_names, bam_names
        list context
        bint use_bq, use_mq, use_dend, use_context, use_userdefined, use_bam
        bint use_contamination
        public int rowlen
        cdef double[:,:] ccontam_rows
        cdef double[:,:] crevcomp_contam_rows
        cdef dict contextindices
        cdef object contam_rows
        cdef object revcomp_contam_rows
