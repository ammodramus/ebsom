import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import scipy.optimize as opt
import h5py
import numpy as np
import numpy.random as npr
import afd
import gradient
import cygradient
import cylikelihood
import beta_with_spikes_integrated as bws
import util
import sys
import argparse
import datetime
from scipy.special import logsumexp
import cyutil as cut
import numpy.random as npr

import neural_net as nn

import h5py_util

parser = argparse.ArgumentParser(
        description='split error modeling files into training and target',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('input', help = 'input HDF5 file')
parser.add_argument('params',
        help = 'file(s) with parameters, one per line. do not include allele-frequency distribution parameters',
        nargs = '+')
parser.add_argument('--num-loci', type = int, help = 'randomly sample this many loci from each bam')
parser.add_argument('--seed', type = int, help = 'random seed for using same loci across runs')
parser.add_argument('--neural-net', action = 'store_true', help = 'specify if parameters are for neural net')
parser.add_argument('--hidden-layer-sizes', type = int, nargs = '+', help = 'number of hidden layers in neural net')
args = parser.parse_args()

if args.seed is not None:
    npr.seed(args.seed)

print '#' + ' '.join(sys.argv)

dat = h5py.File(args.input, 'r')
print '# loading all_majorminor'
all_majorminor = h5py_util.get_major_minor(dat)
print '# obtaining column names'
colnames_str = dat.attrs['covariate_column_names']
colnames = colnames_str.split(',')

h5cm = dat['covariate_matrices']
h5lo = dat['locus_observations']
h5mm = dat['major_minor']

regkeys = [(b, r) for b in 'ACGT' for r in (1,2)]
rowlen = dat.attrs['rowlen']
blims = {}
for i, reg in enumerate(regkeys):
    low = rowlen*3*i
    high = rowlen*3*(i+1)
    blims[reg] = (low, high)

nbetas = len(regkeys)*3*rowlen
pars = []
for parfile in args.params:
    pars.append(np.loadtxt(parfile))

num_pf_params = 3
betas = []
for parset in pars:
    betas.append(parset[:-num_pf_params])

column_names = []
column_names.extend(['direction', 'major', 'minor', 'rmajor', 'rminor', 'chrom', 'bam', 'locus'])
for param_idx, param in enumerate(args.params):
    for regkey in regkeys:
        regrep = str(regkey[0]) + str(regkey[1])
        for outcome in 'ACGT':
            col = '_'.join([str(param_idx), regrep, outcome])
            column_names.append(col)
print '\t'.join(column_names)

# TODO write header with columns

for chrom in h5cm.keys():
    for bam in h5cm[chrom].keys():
        tmm = h5mm[chrom][bam][:]
        if args.num_loci and args.num_loci < len(h5cm[chrom][bam].keys()):
            sampled_loci = npr.choice(h5cm[chrom][bam].keys(), size = args.num_loci, replace = False)
        else:
            sampled_loci = h5cm[chrom][bam].keys()
        for locus in sampled_loci:
            tcm = h5cm[chrom][bam][locus][:]
            if not args.neural_net:
                X = tcm
                logprobs = []
                for betaset in betas:
                    tlogprobs = {}
                    for reg in regkeys:
                        low, high = blims[reg]
                        b = betaset[low:high].reshape((rowlen,-1), order = 'F')
                        Xb = np.column_stack((np.dot(X,b), np.zeros(X.shape[0])))
                        Xb -= logsumexp(Xb, axis = 1)[:,None]
                        tlogprobs[reg] = Xb
                    logprobs.append(tlogprobs)
            else:   # args.neural_net
                # for each parameter set (betaset, need to change):
                #     for each 'regression' ('A',1), ('A',2), etc..., form the
                #     cm matrix, get the logprobs, store in the logprobs for the parameter set
                logprobs = []
                for betaset in betas:  # these are all of the neural net parameters
                    tlogprobs = {}
                    for reg in regkeys:
                        # construct the cm with the true allele and read number (ie the 'reg')
                        base_columns = np.zeros((tcm.shape[0], 4))
                        base_columns[:,'ACGT'.index(reg[0])] = 1.0
                        readtwos = np.ones(tcm.shape[0]) * (reg[1] == 2)
                        regcm = np.column_stack((base_columns, readtwos, tcm))
                        # get matrices
                        num_inputs = regcm.shape[1]
                        num_obs = regcm.shape[0]
                        matrices, num_params = nn.set_up_neural_net(num_inputs, args.hidden_layer_sizes, num_obs)
                        tlogprobs[reg] = nn.neural_net_logprobs_wrapper(betaset, regcm.T, matrices).T
                    logprobs.append(tlogprobs)

            loc_int = int(locus)
            major, minor = tmm[loc_int,:]
            major, minor = str(major), str(minor)
            rmajor, rminor = cut.comp(major), cut.comp(minor)
            majorminors = [major, minor, rmajor, rminor]
            context = [chrom, bam, locus]
            for direc in 'fr':
                for rn in '12':
                    dr = direc + rn
                    tlo = h5lo[chrom][bam][locus][dr][:]
                    if 0 in tlo.shape:
                        continue
                    for obsidx in tlo[:,0]:
                        tlp = []
                        for tlogprobs in logprobs:
                            for reg in regkeys:
                                tlp.extend(list(tlogprobs[reg][obsidx,:]))
                        res = [str(el) for el in [dr] + majorminors + context + tlp]
                        print '\t'.join(res)
