cimport numpy as np
from libc.stdint cimport *

cdef class RegCov:
    cdef:
        float *X
        uint32_t *y
        public int ncol
        public int maxnrows
        public int curnrows
        public double growby
        public dict H   # maps row to index in X, if stored
    cpdef int set_default(self, np.ndarray[ndim=1,dtype=np.float32_t] x_np, uint8_t base_idx)
