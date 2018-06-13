cimport numpy as np
from libc.stdlib cimport malloc, free, realloc

np.import_array()

cdef class LocObs(object):
    def __init__(self, int init_size = 200, double growby = 1.5):
        self.c = <int *>malloc(sizeof(int) * 5 * init_size + 5)
        if not self.c:
            raise MemoryError('could not allocate X')
        self.curnrows = 0
        self.maxnrows = init_size
        self.growby = growby
        if self.growby <= 1.0:
            raise ValueError('growby must be > 1')
        
        #cdef unordered_map[int,int] L
        #self.L = L
        self.L = {}
    
    cpdef void add_obs(self, int reg_row_idx, int base_idx):
        cdef:
            int c_idx
            cdef int *thisrow
        if base_idx > 4 or base_idx < 0:
            raise ValueError('invalid base_idx')
        try:
            c_idx = self.L[reg_row_idx]
            thisrow = &(self.c[5*c_idx])
            thisrow[base_idx+1] += 1
        except KeyError:
            c_idx = self.curnrows
            self.L[reg_row_idx] = c_idx
            self.curnrows += 1
            if self.curnrows > self.maxnrows:
                newnrows = <size_t>(self.maxnrows*self.growby + 1)
                newsize = <size_t>(newnrows * 5 * sizeof(int))
                self.c = <int *>realloc(<void *>self.c, newsize)
                if not self.c:
                    raise MemoryError('could not resize c, too big')
                self.maxnrows = newnrows
            thisrow = &(self.c[5*c_idx])
            for i in range(5):  # set the row
                thisrow[i] = 0
            thisrow[0] = reg_row_idx
            thisrow[base_idx+1] = 1

    cpdef np.ndarray counts(self):
        cdef np.npy_intp dims[2]
        dims[0] = self.curnrows
        dims[1] = 5
        ret = np.PyArray_SimpleNewFromData(2, dims, np.NPY_INT, <void *>self.c)
        return ret

    def __repr__(self):
        return str(self.counts())

    def __dealloc__(self):
        free(self.c)
