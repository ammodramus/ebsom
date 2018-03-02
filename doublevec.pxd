cdef class DoubleVec:
    cdef:
        double *data
        double grow_by
        double cur_max
        unsigned int size
        unsigned int max_size
    cdef inline void append(self, double v)
    cdef inline void clear(self)
