import numpy as np
import numpy.random as npr
import scipy.optimize as opt
import sys
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import h5py
import util
import argparse

parser = argparse.ArgumentParser(
        description='convert collapsed covariate matrices to expanded',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('input', help = 'input HDF5 file')
parser.add_argument('output', help = 'output HDF5 file')
args = parser.parse_args()

inf = h5py.File(args.input, 'r')
out = h5py.File(args.output, 'w')


origcm = inf['covariate_matrix'][:,:]


rowlen = origcm.shape[1]
cm, cm_minmaxes = util.normalize_covariates(origcm)


out.attrs['rowlen'] = rowlen
out.create_dataset('covariates_mins_maxes', data = cm_minmaxes)
out.copy(inf['major_minor'], 'major_minor')


outcm = out.create_group('covariate_matrices')
outlo = out.create_group('locus_observations')


for chrom in inf['locus_observations'].keys():
    chromcm = outcm.create_group(chrom)
    chromlo = outlo.create_group(chrom)
    for bam in inf['locus_observations'][chrom].keys():
        bam_idx = inf['locus_observations'][chrom].keys().index(bam)
        bams_len = len(inf['locus_observations'][chrom].keys())
        bamcm = chromcm.create_group(bam)
        bamlo = chromlo.create_group(bam)
        for locus in inf['locus_observations'][chrom][bam].keys():
            print '\r{} ({} of {}) {}'.format(bam, bam_idx, bams_len, locus),
            locuslo = bamlo.create_group(locus)
            reg_cm = []
            cur_cm_idx = 0
            for reg in ['f1','f2','r1','r2']:
                locobs = inf['locus_observations'][chrom][bam][locus][reg][:,:]
                reg_idxs = []
                for idx in locobs[:,0]:
                    reg_cm.append(cm[idx])
                    reg_idxs.append(cur_cm_idx)
                    cur_cm_idx += 1
                loclodat = np.column_stack((reg_idxs, locobs[:,1:]))
                locuslo.create_dataset(reg, shape = (locobs.shape[0], 5), dtype = np.uint32, data = loclodat, compression = 'gzip', compression_opts = 7)
            reg_cm = np.array(reg_cm)
            bamcm.create_dataset(locus, data = reg_cm, dtype = np.float64, compression = 'gzip', compression_opts = 7)

out.attrs['covariate_column_names'] = inf.attrs['covariate_column_names']


out.close()
inf.close()
