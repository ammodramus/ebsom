from __future__ import division, print_function
import argparse
import numpy as np
import pysam
from locuscollectors import NonCandidateCollector, CandidateCollector
import util as ut
import cyprocessreads as cpr
import cyregression as cre

desc = 'jointly infer sequencing error profiles and polymorphisms'
parser = argparse.ArgumentParser(
        description=desc,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('bams', help = 'file containing list of bam files')
parser.add_argument('references',
        help = 'name of file containing list of reference sequence names, or '
               'comma-separated list of reference sequence names')
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
parser.add_argument('--save-data-as', help = 'filename for optional HDF5 data output')
parser.add_argument('--load-data-from',
        help = 'load data from previously saved HDF5 file')
parser.add_argument('--no-mapq', action = 'store_true',
        help = 'do not use map qualities')
parser.add_argument('--no-bam', action = 'store_true',
        help = 'do not add dummy variable for bam')
parser.add_argument('--min-coverage', type = int, default = 20)
args = parser.parse_args()
min_bq = args.min_bq
min_mq = args.min_mq
context_len = args.context_length

# break bam names into prefix/name
prefix, bam_fns, bams = ut.get_bams(args.bams)
ref_names = ut.get_ref_names(args.references, bams)

print('getting counts')
all_counts = ut.get_all_counts(bams, ref_names, min_bq)
print('getting freqs')
all_freqs = ut.get_freqs(all_counts)
print('getting consensus')
all_consensuses = ut.get_all_consensuses(all_counts, min_coverage = args.min_coverage)
print('getting major-minor')
all_majorminor = ut.get_all_majorminor(all_counts)


# make rowmakers: a dict by ref and bam_fn
row_makers, rowlen = ut.get_row_makers(bam_fns, ref_names, context_len, 
        args.round_distance_by, all_consensuses, not args.no_mapq, not args.no_bam)

if args.load_data_from is None:
    covariate_matrices = ut.get_covariate_matrices(rowlen)
    locus_observations = ut.get_locus_observations(all_majorminor)
                
    for ref in ref_names:
        for bam_fn in bam_fns:
            reflen = len(all_consensuses[ref][bam_fn])
            bam = bams[bam_fn]
            counts = all_counts[ref][bam_fn]
            freqs = all_freqs[ref][bam_fn]
            rm = row_makers[ref][bam_fn]
            mm = all_majorminor[ref][bam_fn]
            locobs = locus_observations[ref][bam_fn]
            consensus = all_consensuses[ref][bam_fn]
            cpr.add_bam_observations(bam, ref, reflen, min_bq, min_mq, context_len,
                    rm, bam_fn, consensus, covariate_matrices, locobs, mm)

    cm = ut.collect_covariate_matrices(covariate_matrices)
    lo = ut.collect_loc_obs(locus_observations)
    # they're all the same...
    cm_names = row_makers[ref][bam_fn].get_covariate_names()
            
    if args.save_data_as is not None:
        try:
            import deepdish as dd
        except ImportError:
            raise ImportError('saving output requires deepdish')
        data = (cm, lo, all_majorminor, cm_names)

        import warnings
        with warnings.catch_warnings():
            dd.io.save(args.save_data_as, data)

else:
    try:
        import deepdish as dd
    except ImportError:
        raise ImportError('saving output requires deepdish')
    cm, lo, all_majorminor = dd.io.load(args.load_data_from)
