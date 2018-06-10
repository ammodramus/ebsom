from libc.stdlib cimport malloc, free

cdef bytes COMPBASES=b'ACGTN'
cdef bytes COMPRCBASES=b'TGCAN'
cdef char* cCOMPBASES = COMPBASES
cdef char* cCOMPRCBASES = COMPRCBASES
cpdef bytes comp(bytes seq):
    cdef:
        char* cseq
        char* crc
        size_t seq_len, i, j
        bint found
        bytes ret

    cseq = seq
    seq_len = len(seq)
    crc = <char *>malloc(sizeof(char) * (seq_len+1))
    if crc == NULL:
        raise MemoryError('out of memory')
    for i in range(seq_len):
        found = False
        for j in range(5):
            if cseq[i] == cCOMPBASES[j]:
                crc[i] = cCOMPRCBASES[j]
                found = True
                break
        if not found:
            raise ValueError('invalid base in {}'.format(seq))
    crc[seq_len] = '\0'
    ret = crc  # makes a copy
    free(crc)
    return ret
