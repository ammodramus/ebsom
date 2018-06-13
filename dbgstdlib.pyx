from libc.stdlib cimport malloc, free, realloc, calloc

cdef void* dbgmalloc(size_t size, bytes desc):
    cdef void *ptr = malloc(size)
    print 'allocating {} bytes at {}'.format(int(size), desc)
    return ptr

cdef void* dbgrealloc(void *ptr, size_t size, bytes desc):
    print 'reallocating to new size {} at {}'.format(size, desc)
    return realloc(ptr, size)

cdef void dbgfree(void *ptr, bytes desc):
    print 'freeing at {}'.format(desc)
    free(ptr)

cdef void* dbgcalloc(size_t nmemb, size_t size, bytes desc):
    print 'c-allocating {} bytes at {}'.format(size, desc)
    return calloc(nmemb, size)
