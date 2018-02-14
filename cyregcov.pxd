cimport numpy as np

cdef class RegCov:
    cdef:
        double *X
        public int ncol
        public int maxnrows
        public int curnrows
        public double growby
        public dict H   # maps row to index in X, if stored
    cpdef int set_default(self, np.ndarray[ndim=1,dtype=np.float64_t] x_np)
