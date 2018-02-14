cimport numpy as np

cdef class LocObs:
    cdef:
        int *c
        public int maxnrows
        public int curnrows
        public double growby
        dict L
    cpdef void add_obs(self, int reg_row_idx, int base_idx)
    cpdef np.ndarray counts(self)
