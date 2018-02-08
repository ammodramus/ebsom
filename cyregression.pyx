cimport numpy as np
import numpy as np
from collections import defaultdict
from scipy.special import logsumexp
import scipy
import time

cdef void add_numerators(
    np.ndarray[ndim=2,dtype=np.float64_t] xdotb,
    np.ndarray[ndim=1,dtype=np.float64_t] loclls,
    np.ndarray[ndim=2,dtype=np.long_t] obs):
    
    cdef:
        int obs_i, cur_idx, col, nobs
    
    cur_idx = 0
    for obs_i in range(obs.shape[0]):
        col, nobs = obs[obs_i,0], obs[obs_i,1]
        if col != -1:
            for i in range(nobs):
                loclls[cur_idx] += xdotb[cur_idx,col]
                cur_idx += 1
        else:
            cur_idx += nobs

cdef class Regression(object):
    cdef:
        public int ncol
        public int blen
        public dict nc_obs   # obs[reg][obase] is the tuple (X,c) of non-candidate
                             # observations for reg, obase
        public dict nc_cov
        public dict nc_count
        public dict nc_cons
        public list list_nc_cov
        public list list_nc_count
        public list list_nc_obs
        public list list_nc_cons
        public list b_coords
        public list regs
        int nregs
        
        
    def __init__(self, nco):
        self.nc_obs = {}
        ncol = None
        self.blen = 0
        self.list_nc_cov = []
        self.list_nc_count = []
        self.list_nc_obs = []
        self.list_nc_cons = []
        self.b_coords = []
        self.nc_obs = {}
        self.nc_cov = {}
        self.nc_count = {}
        self.nc_cons = {}
        self.regs = []
        for key, (X, c) in nco.iteritems():
            if ncol is None:
                ncol = X.shape[1]
                self.ncol = ncol
            else:
                if X.shape[1] != ncol:
                    raise ValueError('covariate matrix wrong shape')
            regression_key = key[:2]
            cons_base = regression_key[0]
            obs_base = key[2]
            other_bases = cons_base + ''.join(el for el in list('ACGT') if el != cons_base)
            obs_idx = other_bases.index(obs_base) - 1  # -1 for consensus base
            cons_idx = 'ACGT'.index(cons_base)
            obs = (obs_idx, c.shape[0])
            cons = (cons_idx, c.shape[0])
            if regression_key not in self.nc_obs:
                self.nc_obs[regression_key] = []
                self.nc_cov[regression_key] = []
                self.nc_count[regression_key] = []
                self.nc_cons[regression_key] = []
            self.nc_obs[regression_key].append(obs)
            self.nc_cov[regression_key].append(X)
            self.nc_count[regression_key].append(c)
            self.nc_cons[regression_key].append(cons)
        
        assert self.nc_obs.keys() == self.nc_cov.keys()
        cur_b_coord = 0
        self.nregs = 0
        for reg in self.nc_obs.keys():
            self.list_nc_cov.append(np.row_stack(self.nc_cov[reg]))
            self.list_nc_count.append(np.concatenate(self.nc_count[reg]))
            self.list_nc_obs.append(np.row_stack(self.nc_obs[reg]))
            self.list_nc_cons.append(np.row_stack(self.nc_cons[reg]))
            self.b_coords.append((cur_b_coord, cur_b_coord + 3*self.ncol))
            cur_b_coord += 3*self.ncol
            self.nregs += 1
            self.regs.append(reg)
        self.blen = cur_b_coord
        
            
    def loglike(self, np.ndarray[ndim=1,dtype=np.float64_t] npbetas):
        cdef:
            double[:] betas = npbetas
            int nbetas = npbetas.shape[0]
            int i, low, high, cidx, cons_idx
            double ll = 0
            np.ndarray[ndim=2,dtype=np.float64_t] br
            
        if nbetas != self.blen:
            raise ValueError('incorrect betas length')
            
        ll = 0
        for i in range(self.nregs):
            X = self.list_nc_cov[i]
            c = self.list_nc_count[i]
            obs = self.list_nc_obs[i]
            cons = self.list_nc_cons[i]
            low, high = self.b_coords[i]
            b = npbetas[low:high]
            br = np.reshape(b, (self.ncol, 3), order = 'F')
            xdotb = np.dot(X,br)
            # add the zeros for simpler logsumexp calculation... not a rate-limiting step.
            xdotb = np.column_stack((xdotb, np.zeros(X.shape[0])))
            loclls = -1 * scipy.special.logsumexp(xdotb, axis = 1)
            add_numerators(xdotb, loclls, obs)
            ll += np.sum(c*loclls)
        return ll
    
    
    def gradient(self, np.ndarray[ndim=1,dtype=np.float64_t] npbetas):
        cdef:
            int i, j, k, l, n
            np.ndarray[ndim=1,dtype=np.float64_t] allgrad
        
        allgrad = np.zeros(self.blen)
        for i in range(self.nregs):
            X = self.list_nc_cov[i]
            c = self.list_nc_count[i]
            obs = self.list_nc_obs[i]
            cons = self.list_nc_cons[i]
            low, high = self.b_coords[i]
            #print('low, high:', low, high)
            #print(X.flags.f_contiguous)
            ymp = np.transpose(self._get_y_matrix(i) - self._get_p_outcomes(npbetas, i))
            grad = np.dot(ymp, X*c[:,None])
            allgrad[low:high] = grad[:3,:].flatten(order='C')
        return allgrad
    
    
    def _get_y_matrix(self, reg_idx):
        cdef:
            np.ndarray[ndim=2,dtype=np.long_t] obs_np
            np.ndarray[ndim=2,dtype=np.float64_t] y_np
            long[:,:] obs
            double[:,:] y
            int i, Xs0, nobs, cur_idx, obs_i, col
            
        obs_np = self.list_nc_obs[reg_idx]
        obs = obs_np
        Xs0 = self.list_nc_cov[reg_idx].shape[0]  # 0th dimension of X
        y_np = np.zeros((Xs0, 4), dtype = np.float64)
        y = y_np
        cur_idx = 0
        for obs_i in range(obs_np.shape[0]):
            col, nobs = obs[obs_i,0], obs[obs_i,1]
            if col != -1:
                for i in range(nobs):
                    y[cur_idx, col] = 1.0
                    cur_idx += 1
            else:
                cur_idx += nobs
        return y_np
    
    
    def _get_p_outcomes(self, betas, i):
        X = self.list_nc_cov[i]
        c = self.list_nc_count[i]
        obs = self.list_nc_obs[i]
        cons = self.list_nc_cons[i]
        low, high = self.b_coords[i]
        b = betas[low:high]
        br = np.reshape(b, (self.ncol, 3), order = 'F')
        xdotb = np.dot(X,br)
        # add the zeros for simpler logsumexp calculation... not a rate-limiting step.
        xdotb = np.column_stack((xdotb, np.zeros(X.shape[0])))
        lse = scipy.special.logsumexp(xdotb, axis = 1)
        p_outcomes = np.exp(xdotb-lse[:,None])
        return p_outcomes
