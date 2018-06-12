## cython: profile=True
## cython: linetrace=True
## cython: binding=True
## distutils: define_macros=CYTHON_TRACE_NOGIL=1

from libc.stdlib cimport malloc, free
from libc.string cimport memcpy
cimport numpy as np
import numpy as np

from cpython cimport Py_INCREF
from libc.stdlib cimport malloc, free, realloc

np.import_array()

'''
tried to use super-fast hash function to speed up hashing unique rows,
but not as fast as using x_np.to_bytes(). may be because of Python overhead in
ArrKey.__eq__
'''
cdef extern from "xxHash/xxhash.c":
    unsigned long long XXH64(const void* buf, size_t length, unsigned long long seed)

cdef unsigned long long get_hash(const void *buf, size_t length):
    cdef unsigned long long seed = 0;
    cdef unsigned long long hashnum = XXH64(buf, length, seed)
    return hashnum

cdef class ArrKey:
    cdef double *data
    cdef public unsigned int length
    cdef public object arr
    def __init__(self, np.ndarray[ndim=1,dtype=np.float64_t] data):
        self.data = <double *>&(data.data[0])
        self.length = data.shape[0]
        self.arr = data

    def __hash__(self):
        ret = get_hash(<void *>self.data, <size_t>(self.length*sizeof(double)))
        return ret

    def __eq__(self, other):
        cdef int i
        cdef np.ndarray[ndim=1,dtype=np.float64_t] otherarr = other.arr
        cdef double *otherdat = &otherarr[0]
        for i in range(self.length):
            if self.data[i] != otherdat[i]:
                return False
        return True



cdef class RegCov:
    def __init__(self, int ncol, int init_size = 10000, double growby = 1.5):
        self.X = <double *>malloc(sizeof(double) * ncol * init_size + ncol)
        if not self.X:
            raise MemoryError('could not allocate X')
        self.curnrows = 0
        self.maxnrows = init_size
        self.ncol = ncol
        self.growby = growby
        if self.growby <= 1.0:
            raise ValueError('growby must be > 1')
        self.H = {}
    
    cpdef int set_default(self, np.ndarray[ndim=1,dtype=np.float64_t] x_np):
        cdef int ret, i, newnrows
        cdef size_t newsize
        cdef double *thisrow
        cdef double *x
        x = &x_np[0]
        #cdef ArrKey key = ArrKey(x_np)
        cdef bytes key = x_np.tobytes()
        ret = self.H.get(key, -1)
        if ret == -1:
            ret = self.curnrows
            self.H[key] = ret
            self.curnrows += 1
            if self.curnrows > self.maxnrows:
                newnrows = <size_t>(self.maxnrows*self.growby + 1)
                newsize = <size_t>(newnrows * self.ncol * sizeof(double))
                self.X = <double *>realloc(<void *>self.X, newsize)
                if not self.X:
                    raise MemoryError('could not resize X, too big')
                self.maxnrows = newnrows
            thisrow = &(self.X[self.ncol*ret])
            for i in range(self.ncol):  # set the row
                thisrow[i] = x[i]
        return ret
    
    def covariate_matrix(self):
        cdef np.npy_intp dims[2]
        dims[0] = self.curnrows
        dims[1] = self.ncol
        ret = np.PyArray_SimpleNewFromData(2, dims, np.NPY_FLOAT64, <void *>self.X)
        return ret

    def __dealloc__(self):
        free(self.X)
