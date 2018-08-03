from __future__ import division
import h5py
import numpy as np
import numpy.random as npr
import sys
import argparse


def get_major_minor(h5in):
    mm = {}
    for chrom in h5in['major_minor'].keys():
        h5_chrom_mm = h5in['major_minor'][chrom]
        mm[chrom] = {}
        for bam in h5_chrom_mm.keys():
            h5_bam_mm = h5_chrom_mm[bam]
            t_h5_bam_mm = h5_bam_mm[:,:].copy()
            mm[chrom][bam] = t_h5_bam_mm
    return mm


print '#' + ' '.join(sys.argv)


parser = argparse.ArgumentParser(
        description='optimize neural-net-based model of heteroplasmy \
                and sequencing error',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('input', help = 'input HDF5 file')
parser.add_argument('trainout', help = 'training output HDF5 file')
parser.add_argument('testout', help = 'testing output HDF5 file')
parser.add_argument('--frac-test', type = float, default = 0.1)
parser.add_argument('--bad-locus-file')
parser.add_argument('--seed', help = 'random seed', type = int)
args = parser.parse_args()

if args.seed is not None:
    npr.set_seed(args.seed)

dat = h5py.File(args.input, 'r')
print '# loading all_majorminor'
all_majorminor = get_major_minor(dat)
print '# obtaining column names'
colnames_str = dat.attrs['covariate_column_names']
colnames = colnames_str.split(',')

h5cm = dat['covariate_matrices']
h5lo = dat['locus_observations']

print '# getting covariate matrix / locus observation keys'
h5cm_keys = []
for chrom, chrom_cm in h5cm.iteritems():
    for bam, bam_cm in chrom_cm.iteritems():
        print '# getting covariate matrix keys: {}'.format(bam)
        for locus, locus_cm in bam_cm.iteritems():
            spname = locus_cm.name.split('/')
            name = unicode('/'.join(spname[2:]))
            h5cm_keys.append(name)

locus_keys = h5cm_keys

# badloci.txt is 1-based, these will be 0-based. notice chrM is hard-coded
if args.bad_locus_file is not None:
    badloci = set(list(np.loadtxt(args.bad_locus_file).astype(np.int)-1))
    good_keys = []
    for key in locus_keys:
        locus = int(key.split('/')[-1])
        if locus not in badloci:
            good_keys.append(key)
else:
    good_keys = h5cm_keys

trainout = h5py.File(args.trainout, 'w')
testout = h5py.File(args.testout, 'w')

shuffled_keys = npr.permutation(good_keys)
num_test = int(len(good_keys)*args.frac_test + 0.5)
num_train = len(shuffled_keys) - num_test

if num_test == 0 or num_train == 0:
    raise ValueError('not enough training or testing keys: {} training, {} testing'.format(num_train, num_test))

train_keys = shuffled_keys[:num_train]
test_keys = shuffled_keys[num_train:]
import pdb; pdb.set_trace()

cmtrainout = trainout.create_group('covariate_matrices')
cmtestout = testout.create_group('covariate_matrices')

lotrainout = trainout.create_group('locus_observations')
lotestout = testout.create_group('locus_observations')

for key in train_keys:
    cmtrainout.copy(h5cm[key], key)
    lotrainout.copy(h5lo[key], key)

for key in test_keys:
    cmtestout.copy(h5cm[key], key)
    lotestout.copy(h5lo[key], key)

trainout.copy(dat['major_minor'], 'major_minor')
testout.copy(dat['major_minor'], 'major_minor')

for key in dat.attrs.keys():
    trainout.attrs[key] = dat.attrs[key]
    testout.attrs[key] = dat.attrs[key]
