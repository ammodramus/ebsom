import numpy as np
import deepdish as dd
from sys import getsizeof
from itertools import chain
from collections import deque
try:
    from reprlib import repr
except ImportError:
    pass

def total_size(o, handlers={}, verbose=False):
    dict_handler = lambda d: chain.from_iterable(d.items())
    all_handlers = {tuple: iter,
                    list: iter,
                    deque: iter,
                    dict: dict_handler,
                    set: iter,
                    frozenset: iter,
                   }
    all_handlers.update(handlers)     # user handlers take precedence
    seen = set()                      # track which object id's have already been seen
    default_size = getsizeof(0)       # estimate sizeof object without __sizeof__

    def sizeof(o):
        if id(o) in seen:       # do not double count the same object
            return 0
        seen.add(id(o))
        s = getsizeof(o, default_size)
        return s

    return sizeof(o)

cm, lo, all_majorminor, colnames = dd.io.load('data_TR10.h5')
nrows = 0
for chrom in lo.keys():
    for bam in lo[chrom].keys():
        for pos, pos_lo in enumerate(lo[chrom][bam]):
            for i in range(2):
                for j in range(2):
                    tnrows = pos_lo[i][j].shape[0]
                    ncols = pos_lo[i][j].shape[1]
                    nrows += tnrows

all_lo = np.zeros((nrows, ncols), dtype = np.int32)
indices = {}
nrows = 0
cur_idx = 0
for chrom in lo.keys():
    indices[chrom] = {}
    for bam in lo[chrom].keys():
        indices[chrom][bam] = []
        for pos, pos_lo in enumerate(lo[chrom][bam]):
            indices[chrom][bam].append([])
            for i in range(2):
                indices[chrom][bam][-1].append([])
                for j in range(2):
                    tnrows = pos_lo[i][j].shape[0]
                    ncols = pos_lo[i][j].shape[1]
                    indices[chrom][bam][-1][-1].append((cur_idx, cur_idx + tnrows))
                    all_lo[cur_idx:cur_idx+tnrows,:] = pos_lo[i][j]
                    cur_idx += tnrows

data = (cm, all_lo, all_majorminor, colnames)

import warnings
with warnings.catch_warnings():
    dd.io.save('data_TR10_restructured_locobs.h5', data)
