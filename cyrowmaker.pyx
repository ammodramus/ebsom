import numpy as np

cdef class CovariateRowMaker(object):
    cdef:
        char consbase
        int contextlen
        int dend_roundby
        str thiscons
        list othercons
        list userdefined_names
        bint use_bq, use_mq, use_dend, use_context, use_userdefined

    def __init__(self,
                 char consbase,
                 int contextlen,
                 int dend_roundby,
                 str thiscons = None,
                 str othercons = None,
                 userdefined = None,
                 userdefined_names = None,
                 bint use_bq = True,
                 bint use_mq = True,
                 bint use_dend = True):
        self.consbase = consbase
        self.contextlen = contextlen
        self.dend_roundby = dend_roundby
        self.thiscons = thiscons
        self.othercons = othercons
        self.userdefined = userdefined
        self.userdefined_names = userdefined_names
        self.use_bq = use_bq
        self.use_mq = use_mq
        self.use_dend = use_dend
        self.use_context = self.contextlen >= 1
        self.use_userdefined = userdefined is not None
        
        if thiscons is not None and othercons is not None:
            self.use_contamination = True
        else:
            self.use_contamination = False
        
        if self.contextlen is not None and self.contextlen > 0:
            bases = 'ACGT'
            c = ['']
            for i in xrange(contextlen):
                cp = []
                for cel in c:
                    for b in bases:
                        cp.append(cel + b)
                c = cp
            refcontext = consbase * contextlen
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
                    all_ncols = userdefined[key].shape[0]
                else:
                    this_ncol = userdefined[key].shape[0]
                    if this_ncol != all_ncols:
                        raise ValueError('different array lengths in userdefined dict')
            self.user_ncols = all_ncols
        
        if self.use_contamination:
            thiscons = np.array(list(thiscons))
            othercons = np.array([np.array(list(oc)) for oc in othercons])
            contam_rows = np.zeros((thiscons.shape[0], 4))
            #otherbases = [b for b in 'ACGT' if b != self.consbase]
            otherbases = 'ACGT'
            for i, ob in enumerate(otherbases):
                countdiffs = np.zeros_like(thiscons, dtype = np.float64)
                countcalls = np.zeros_like(thiscons, dtype = np.float64)
                for oc in othercons:
                    countcalls += (oc != 'N')
                    filt = ((oc != 'N') & (thiscons != 'N') & (oc != thiscons)
                            & (oc == ob))
                    countdiffs[filt] += 1
                frac_contam = countdiffs / np.maximum(countcalls, 1)
                contam_rows[:,i] = frac_contam
            self.contam_rows = contam_rows
            self.revcomp_contam_rows = contam_rows[:,::-1].copy()
        self.rowlen = (1 + use_bq + use_mq + use_dend + 
                4*self.use_contamination + (4**self.contextlen) -
                1 + self.user_ncols)
    
    def get_covariate_row(self,
            consbase, bq, mq, context, dend, refpos, bam_name, reverse):
        row = np.zeros(self.rowlen)
        row[0] = 1.0   # constant
        cur_idx = 1
        if self.use_bq:
            row[cur_idx] = bq
            cur_idx += 1
        if self.use_mq:
            row[cur_idx] = mq
            cur_idx += 1
        if self.use_dend:
            row[cur_idx] = ((dend + (self.dend_roundby)/2.0)
                    // self.dend_roundby * self.dend_roundby)
            cur_idx += 1
        if self.use_contamination:
            if not reverse:
                row[cur_idx:cur_idx+4] = self.contam_rows[refpos]
            else:
                row[cur_idx:cur_idx+4] = self.revcomp_contam_rows[refpos]
            cur_idx += 4
        if self.use_context:
            row[cur_idx:cur_idx+self.ncontexts] = 0
            try:
                context_index = self.contextindices[context]
                row[cur_idx + context_index] = 1.0
            except:
                pass
            cur_idx += self.ncontexts
        if self.use_userdefined:
            row[cur_idx:cur_idx+self.user_ncols] = self.userdefined_rows[bam_name]
            cur_idx += self.user_ncols
        if cur_idx != self.rowlen:
            raise ValueError('row of incorrect length')
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
        return names
