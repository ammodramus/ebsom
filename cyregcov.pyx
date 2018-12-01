## cython: profile=True
## cython: linetrace=True
## cython: binding=True
## distutils: define_macros=CYTHON_TRACE_NOGIL=1

from __future__ import print_function
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy
cimport numpy as np
import numpy as np

from cpython cimport Py_INCREF
from libc.stdlib cimport malloc, calloc, free, realloc
from libc.string cimport memset
from libc.stdint cimport *

np.import_array()

cdef class RegCov:
    def __init__(self, int ncol, int init_size = 10000, double growby = 1.5):
        self.X = <float *>calloc(ncol*init_size+ncol, sizeof(float))
        self.y = <uint32_t *>calloc(4*(init_size+1), sizeof(uint32_t))
        if not self.X:
            raise MemoryError('could not allocate X')
        if not self.y:
            raise MemoryError('could not allocate y')
        self.curnrows = 0
        self.maxnrows = init_size
        self.ncol = ncol
        self.growby = growby
        if self.growby <= 1.0:
            raise ValueError('growby must be > 1')
        self.H = {}
    
    cpdef int set_default(self, np.ndarray[ndim=1,dtype=np.float32_t] x_np, uint8_t base_idx):
        cdef int ret, i, newnrows
        cdef size_t newsizeX, newsizey
        cdef float *thisrow
        cdef uint32_t *thisrow_y
        cdef float *x
        x = &x_np[0]
        cdef bytes key = x_np.tobytes()
        ret = self.H.get(key, -1)
        if ret == -1:
            ret = self.curnrows
            self.H[key] = ret
            self.curnrows += 1
            if self.curnrows > self.maxnrows:
                newnrows = <size_t>(self.maxnrows*self.growby + 1)
                newsizeX = <size_t>(newnrows * self.ncol * sizeof(float))
                self.X = <float *>realloc(<void *>self.X, newsizeX)
                if not self.X:
                    raise MemoryError('could not resize X, too big')

                newsizey = <size_t>(newnrows * 4 * sizeof(uint32_t))
                self.y = <uint32_t *>realloc(<void *>self.y, newsizey)
                if not self.y:
                    raise MemoryError('could not resize y, too big')

                self.maxnrows = newnrows

            thisrow = &(self.X[self.ncol*ret])
            for i in range(self.ncol):  # set the row
                thisrow[i] = x[i]

            thisrow_y = &(self.y[4*ret])
            for i in range(4):
                thisrow_y[i] = 0  # initialize observations to zero
        thisrow_y = &(self.y[ret*4])
        thisrow_y[base_idx] += 1
        return ret
    
    def covariate_matrix(self):
        cdef np.npy_intp dims[2]
        dims[0] = self.curnrows
        dims[1] = self.ncol
        cdef int i, j
        ret = np.PyArray_SimpleNewFromData(2, dims, np.NPY_FLOAT32, <void *>self.X)
        return ret


    def observations(self):
        cdef np.npy_intp dims[2]
        dims[0] = self.curnrows
        dims[1] = 4  # number of bases
        ret = np.PyArray_SimpleNewFromData(2, dims, np.NPY_UINT32, <void *>self.y)
        return ret

    def __dealloc__(self):
        free(self.X)
        free(self.y)
