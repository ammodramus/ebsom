cimport numpy as np
import numpy as np
cimport cython
import cython

cdef class CyCovariateRowMaker:
    def __init__(self,
                 int contextlen,
                 int dend_roundby,
                 bytes thiscons = None,
                 list othercons = None,
                 list bam_fns = [],
                 userdefined = None,
                 userdefined_names = None,
                 bint use_bq = True,
                 bint use_mq = True,
                 bint use_dend = True,
                 bint use_bam = True):
        self.contextlen = contextlen
        self.dend_roundby = dend_roundby
        self.thiscons = thiscons
        self.othercons = othercons
        self.userdefined_rows = userdefined
        self.userdefined_names = userdefined_names
        self.use_bq = use_bq
        self.use_mq = use_mq
        self.use_dend = use_dend
        self.use_context = self.contextlen >= 1
        self.use_userdefined = userdefined is not None
        self.use_bam = use_bam
        
        if thiscons is not None and othercons is not None:
            self.use_contamination = True
        else:
            self.use_contamination = False
        
        if self.contextlen is not None and self.contextlen > 0:
            bases = 'ACGT'
            c = ['']
            for i in range(contextlen):
                cp = []
                for cel in c:
                    for b in bases:
                        cp.append(cel + b)
                c = cp
            refcontext = 'A' * contextlen
            self.context = [el for el in c if el != refcontext]
            self.contextindices = {el: i for i, el in enumerate(self.context)}
            self.ncontexts = len(self.context)
        
        self.user_ncols = 0
        self.userdefined_rows = None
        if userdefined is not None:
            # userdefined is a dict of numpy arrays, one row per bam file
            self.userdefined_rows = userdefined
            all_ncols = None
            for bam in userdefined.keys():
                if all_ncols is None:
                    all_ncols = userdefined[bam].shape[0]
                else:
                    this_ncol = userdefined[bam].shape[0]
                    if this_ncol != all_ncols:
                        raise ValueError('different array lengths in userdefined dict')
            self.user_ncols = all_ncols
        
        if self.use_contamination:
            pythiscons = np.array(list(thiscons))
            pyothercons = np.array([np.array(list(oc)) for oc in othercons])
            contam_rows = np.zeros((pythiscons.shape[0], 4))
            otherbases = 'ACGT'
            for i, ob in enumerate(otherbases):
                countdiffs = np.zeros_like(pythiscons, dtype = np.float64)
                countcalls = np.zeros_like(pythiscons, dtype = np.float64)
                for oc in othercons:
                    countcalls += (oc != 'N')
                    filt = ((oc != 'N') & (pythiscons != 'N') & (oc != pythiscons)
                            & (oc == ob))
                    countdiffs[filt] += 1
                frac_contam = countdiffs / np.maximum(countcalls, 1)
                contam_rows[:,i] = frac_contam
            self.contam_rows = contam_rows
            self.ccontam_rows = self.contam_rows
            self.revcomp_contam_rows = contam_rows[:,::-1].copy()
            self.crevcomp_contam_rows = self.revcomp_contam_rows

        self.num_bams = 0
        if use_bam:
            self.bam_names = bam_fns
            self.bam_idxs = {}
            for i, bam_fn in enumerate(bam_fns):
                self.bam_idxs[bam_fn] = i-1   # -1 for the first one, will be
                                              # the reference
            self.num_bams = len(bam_fns)
        self.rowlen = (1 + use_bq + use_mq + use_dend + 
                4*self.use_contamination + (4**self.contextlen) -
                1 + self.user_ncols + self.use_bam*(self.num_bams-1))
    
    @cython.boundscheck(False)
    def get_covariate_row(self,
            int bq, int mq, bytes context, int dend, int refpos,
            bytes bam_name, bint reverse):
        cdef:
            int context_index
            np.ndarray[ndim=1,dtype=np.float64_t] row

        row = np.zeros(self.rowlen)
        cdef double[:] crow = row
        crow[0] = 1.0   # constant
        cdef int cur_idx = 1, bam_idx
        if self.use_bq:
            crow[cur_idx] = bq
            cur_idx += 1
        if self.use_mq:
            crow[cur_idx] = mq
            cur_idx += 1
        if self.use_dend:
            crow[cur_idx] = ((dend + (self.dend_roundby)/2.0)
                    // self.dend_roundby * self.dend_roundby)
            cur_idx += 1
        if self.use_contamination:
            if not reverse:
                crow[cur_idx:cur_idx+4] = self.ccontam_rows[refpos,:4]
            else:
                crow[cur_idx:cur_idx+4] = self.crevcomp_contam_rows[refpos,:4]
            cur_idx += 4
        if self.use_context:
            crow[cur_idx:cur_idx+self.ncontexts] = 0
            try:
                context_index = self.contextindices[context]
                crow[cur_idx + context_index] = 1.0
            except:
                pass
            cur_idx += self.ncontexts
        if self.use_userdefined:
            crow[cur_idx:cur_idx+self.user_ncols] = self.userdefined_rows[bam_name]
            cur_idx += self.user_ncols
        if self.use_bam:
            # number of variables is num_bams-1
            crow[cur_idx:cur_idx+self.num_bams-1] = 0
            bam_idx = self.bam_idxs[bam_name]
            if bam_idx >= 0:   # reference bam file encoded as -1
                crow[cur_idx+bam_idx] = 1.0
            cur_idx += self.num_bams-1
        if cur_idx != self.rowlen:
            raise ValueError('row of incorrect length: {}, expected {}'.format(
                cur_idx, self.rowlen))
        row.flags.writeable = False
        return row
        
    def get_covariate_names(self):
        #raise NotImplementedError('covariate names not yet implemented')
        names = ['const']
        if self.use_bq:
            names.append('bq')
        if self.use_mq:
            names.append('mq')
        if self.use_dend:
            names.append('dend')
        if self.use_contamination:
            names.extend(['contam' + b for b in 'ACGT'])
        if self.use_context:
            names.extend(['p'+con for con in self.context])
        if self.use_userdefined:
            names.extend(self.userdefined_names)
        if self.use_bam:
            # first bam name is the reference case, excluded
            names.extend(self.bam_names[1:])
        return names
