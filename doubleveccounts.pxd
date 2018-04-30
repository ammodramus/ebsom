cdef struct doubleveccount_t:
    double *data
    unsigned int *counts
    double grow_by
    unsigned int size
    unsigned int max_size
    double cur_max

cdef void doubleveccount_init(doubleveccount_t *vec, unsigned int init_size, double grow_by)
cdef void doubleveccount_append(doubleveccount_t *vec, double v, unsigned int count)
cdef void doubleveccount_clear(doubleveccount_t *vec)

