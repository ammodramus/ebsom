from libc.stdlib cimport malloc, free, calloc, realloc
from libc.math cimport INFINITY

cdef struct doubleveccount_t:
    double *data
    unsigned int *counts
    double grow_by
    unsigned int size
    unsigned int max_size
    double cur_max

cdef void doubleveccount_init(doubleveccount_t *vec, unsigned int init_size, double grow_by):
    if grow_by <= 1.0:
        raise ValueError('grow_by must be > 1')
    vec.data = <double *>calloc(init_size, sizeof(double))
    vec.counts = <unsigned int *>calloc(init_size, sizeof(unsigned int))
    if vec.data == NULL:
        raise MemoryError('out of memory')
    vec.grow_by = grow_by
    if vec.data == NULL:
        raise MemoryError('out of memory')
    vec.size = 0
    vec.max_size = init_size
    vec.cur_max = -INFINITY

cdef void doubleveccount_append(doubleveccount_t *vec, double v, unsigned int count):
    cdef unsigned int new_max_size
    vec.data[vec.size] = v
    vec.counts[vec.size] = count
    if v > vec.cur_max:
        vec.cur_max = v
    vec.size += 1
    if vec.size >= vec.max_size:
        new_max_size = <unsigned int>(vec.max_size*vec.grow_by + 0.5)
        if new_max_size <= vec.max_size:
            raise ValueError('failed to increase vector size')
        vec.data = <double *>realloc(vec.data, sizeof(double)*new_max_size)
        if not vec.data:
            raise MemoryError('out of memory')
        vec.max_size = new_max_size

cdef void doubleveccount_clear(doubleveccount_t *vec):
    vec.size = 0
    vec.cur_max = -INFINITY
