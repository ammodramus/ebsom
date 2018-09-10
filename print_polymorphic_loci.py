from __future__ import division
import pysam
import h5py
import numpy as np
import argparse

parser = argparse.ArgumentParser(description = 'print out list of chrom/bam/position items where minor allele frequency exceeds threshold --min-freq',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('input', help = 'input HDF5 file')
parser.add_argument('--min-freq', default = 0.01, help = 'minimum frequency to be output', type = float)
args = parser.parse_args()

dat = h5py.File(args.input)
h5lo = dat['locus_observations']
h5cm = dat['covariate_matrices']

min_maf = args.min_freq
bad_keys = []
for chrom in h5lo.keys():
    for bam in h5lo[chrom].keys():
        for locus in h5lo[chrom][bam].keys():
            counts = np.zeros(4)
            for direc in 'fr':
                for readno in '12':
                    tlo = h5lo[chrom][bam][locus][direc+readno][:]
                    if 0 in tlo.shape:
                        continue
                    tlo = tlo[:,1:]
                    if direc == 'r':
                        tlo = tlo[:,::-1]
                    counts += tlo.sum(0)
            total = counts.sum()
            maf = np.sort(counts)[::-1][1] / total
            key = '/'.join([chrom, bam, locus])
            if maf > min_maf:
                print '{}\t{}'.format(key, maf)
            else:
                pass
                #print '# not', key, 'with maf', maf
