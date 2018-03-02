cdef class DoubleVecCounts:
    cdef:
        double *data
        unsigned int *counts
        double grow_by
        double cur_max
        unsigned int size
        unsigned int max_size
    cdef inline void append(self, double v, unsigned int count)
    cdef inline void clear(self)
