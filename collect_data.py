from __future__ import division, print_function
import argparse
import numpy as np
import pysam
from locuscollectors import NonCandidateCollector, CandidateCollector
import util as ut
import cyprocessreads as cpr
import cyregression as cre
import h5py
import h5py_util

desc = 'jointly infer sequencing error profiles and polymorphisms'
parser = argparse.ArgumentParser(
        description=desc,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('bams', help = 'file containing list of bam files')
parser.add_argument('references',
        help = 'name of file containing list of reference sequence names, or '
               'comma-separated list of reference sequence names')
parser.add_argument('output', help = 'filename for optional HDF5 data output')
parser.add_argument('--min-bq', help = 'minimum base quality',
        type = ut.positive_int, default = 20)
parser.add_argument('--min-mq', help = 'minimum mapping quality',
        type = ut.positive_int, default = 20)
parser.add_argument('--min-candidate-freq', type = ut.probability,
        default = 1e-3,
        help = 'minimum frequency at a column to be considered a candidate '
               'for polymorphism')
parser.add_argument('--bam-data',
        help = 'comma-separated (csv) dataset containing supplementary '
               'variables for different bam files. added to covariates for '
               'each observation from a bam file. first column is the bam '
               'name, remaining columns are the covariates. must be '
               'dummy-encoded.')
parser.add_argument('--context-length', type = ut.nonneg_int, default = 2,
        help = 'number of preceding bases to use as covariate')
parser.add_argument('--round-distance-by',
        type = ut.positive_int, default = 10,
        help = 'round distance from start of read by this amount. larger '
               'numbers make for more compression in the data, faster '
               'likelihood evaluations.')
parser.add_argument('--no-mapq', action = 'store_true',
        help = 'do not use map qualities')
parser.add_argument('--no-bam', action = 'store_true',
        help = 'do not add dummy variable for bam')
parser.add_argument('--min-coverage', type = int, default = 20)
parser.add_argument('--do-not-remove-nonvariable', action = 'store_true')
args = parser.parse_args()
min_bq = args.min_bq
min_mq = args.min_mq
context_len = args.context_length

# break bam names into prefix/name
prefix, bam_fns, bams = ut.get_bams(args.bams)
ref_names = ut.get_ref_names(args.references, bams)

print('getting counts')
all_counts = ut.get_all_counts(bams, ref_names, min_bq)
print('getting consensus')
all_consensuses = ut.get_all_consensuses(all_counts, min_coverage = args.min_coverage)
print('getting major-minor')
all_majorminor = ut.get_all_majorminor(all_counts)


# make rowmakers: a dict by ref and bam_fn
row_makers, rowlen = ut.get_row_makers(bam_fns, ref_names, context_len, 
        args.round_distance_by, all_consensuses, not args.no_mapq, not args.no_bam)

covariate_matrices = ut.get_covariate_matrices(rowlen)

output = h5py.File(args.output, 'w')
h5lo = output.create_group('locus_observations')
h5mm = output.create_group('major_minor')

h5py_util.add_major_minor(all_majorminor, h5mm)

for ref in ref_names:
    h5lo_ref = h5lo.create_group(ref)
    for bam_fn in bam_fns:
        h5lo_bam = h5lo_ref.create_group(bam_fn)
        reflen = len(all_consensuses[ref][bam_fn])
        bam = bams[bam_fn]
        rm = row_makers[ref][bam_fn]
        mm = all_majorminor[ref][bam_fn]
        consensus = all_consensuses[ref][bam_fn]
        cpr.add_bam_observations(bam, ref, reflen, min_bq, min_mq, context_len,
                rm, bam_fn, consensus, covariate_matrices, h5lo_bam, output, mm)

# probably best to translate the various lo's to two numpy arrays, one with the
# data, another with the meta data. the names of the bams and refs can be HDF5 attributes

cm = ut.collect_covariate_matrices(covariate_matrices)
# they're all the same...
cm_names = row_makers[ref][bam_fn].get_covariate_names()

if not args.do_not_remove_nonvariable:
    nonvariables = []
    consts = []
    for j in range(cm.shape[1]):
        uniques = np.unique(cm[:,j])
        isnonvar = uniques.shape[0] == 1
        isconst = isnonvar and uniques[0] == 1.0
        nonvariables.append(isnonvar)
        if isconst:
            consts.append(j)

    keeper_columns = []
    keeper_column_names = []
    wrote_const = False
    for j in range(cm.shape[1]):
        write_col = False
        if j in consts:
            if not wrote_const:
                wrote_const = True
                write_col = True
            assert j in nonvariables
        #if j not in nonvariables:
        if not nonvariables[j]:
            write_col = True
        if write_col:
            keeper_columns.append(j)
            keeper_column_names.append(cm_names[j])
    old_cm = cm
    cm = old_cm[:,np.array(keeper_columns)]


output.attrs['covariate_column_names'] = ','.join(list(cm_names))
output.create_dataset('covariate_matrix', dtype = np.float64, data = cm)
