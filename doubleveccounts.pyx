from libc.stdlib cimport malloc, free, calloc, realloc
from libc.math cimport INFINITY

cdef class DoubleVecCounts:

    def __cinit__(self, unsigned int init_size, double grow_by):
        pass

    def __init__(self, unsigned int init_size, double grow_by):
        if grow_by <= 1.0:
            raise ValueError('grow_by must be > 1')
        self.data = <double *>calloc(init_size, sizeof(double))
        self.counts = <unsigned int *>calloc(init_size, sizeof(unsigned int))
        if self.data == NULL or self.counts == NULL:
            raise MemoryError('out of memory')
        self.grow_by = grow_by
        self.size = 0
        self.max_size = init_size
        self.cur_max = -INFINITY

    cdef inline void append(self, double v, unsigned int count):
        cdef unsigned int new_max_size
        self.data[self.size] = v
        self.counts[self.size] = count
        if v > self.cur_max:
            self.cur_max = v
        self.size += 1
        if self.size >= self.max_size:
            new_max_size = <unsigned int>(self.max_size*self.grow_by + 0.5)
            if new_max_size <= self.max_size:
                raise ValueError('failed to increase vector size')
            self.data = <double *>realloc(self.data, sizeof(double)*new_max_size)
            self.counts = <unsigned int *>realloc(self.counts, sizeof(unsigned int)*new_max_size)
            if (not self.data) or (not self.counts):
                raise MemoryError('out of memory')
            self.max_size = new_max_size

    cdef inline void clear(self):
        # note that this doesn't do anything to the data, or the size of the
        # allocated memory
        self.size = 0
        self.cur_max = -INFINITY

    def __dealloc__(self):
        free(self.data)
        free(self.counts)
