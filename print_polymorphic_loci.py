from __future__ import division
import pysam
import h5py
import numpy as np

parser = argparse.ArgumentParser()
parser.add_arguments('input', help = 'input HDF5 file')
parser.add_arguments('--min-freq', default = 0.01, help = 'minimum frequency to be output')

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
            if maf > min_maf:
                key = '/'.join([chrom, bam, locus])
                bad_keys.append(key)

for bk in bad_keys:
    print bk
