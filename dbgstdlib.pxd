cdef void* dbgmalloc(size_t size, bytes desc)
cdef void* dbgrealloc(void *ptr, size_t size, bytes desc)
cdef void dbgfree(void *ptr, bytes desc)
cdef void* dbgcalloc(size_t nmemb, size_t size, bytes desc)
