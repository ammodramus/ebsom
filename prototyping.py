from __future__ import division, print_function
import argparse
import numpy as np
import pysam
from locuscollectors import NonCandidateCollector, CandidateCollector

import util as ut
import processreads as pr
import pyximport; pyximport.install()
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
args = parser.parse_args()
min_bq = args.min_bq
min_mq = args.min_mq
context_len = args.context_length

bam_fns, bams = ut.get_bams(args.bams)
ref_names = ut.get_ref_names(args.references, bams)
print('getting counts')
all_counts = ut.get_all_counts(bams, ref_names, min_bq)
print('getting freqs')
all_freqs = ut.get_freqs(all_counts)
print('getting candidate')
is_candidate = ut.determine_candidates(all_freqs, args.min_candidate_freq)
print('getting consensus')
all_consensuses = ut.get_all_consensuses(all_counts, min_coverage = 20)

# make rowmakers
row_makers, rowlen = ut.get_row_makers(bam_fns, ref_names, context_len, 
        args.round_distance_by, all_consensuses, not args.no_mapq)

if args.load_data_from is None:
    nc_observations = NonCandidateCollector(rowlen)
    c_observations = CandidateCollector(rowlen)
                
    for ref in ref_names:
        for bam_fn in bam_fns:
            reflen = len(all_consensuses[ref][bam_fn])
            bam = bams[bam_fn]
            counts = all_counts[ref][bam_fn]
            freqs = all_freqs[ref][bam_fn]
            is_can = is_candidate[ref][bam_fn]
            rm = row_makers[ref][bam_fn]
            consensus = all_consensuses[ref][bam_fn]
            cpr.add_bam_observations(bam, ref, reflen, min_bq, min_mq, context_len,
                    rm, bam_fn, consensus, is_can, nc_observations, c_observations)
            #i = 0
            #for read in bam.fetch(contig = ref, start = 0, end = reflen):
            #    mapq = read.mapping_quality
            #    if mapq < min_mq:
            #        continue
            #    pr.add_observations(read, mapq, min_bq, context_len, rm,
            #            bam_fn, consensus, is_can, nc_observations, c_observations)
            #    i += 1
            #    if i >= 1000:
            #        break
            
    nco = nc_observations.collect()
    co = c_observations.collect()

    if args.save_data_as is not None:
        try:
            import deepdish as dd
        except ImportError:
            raise ImportError('saving output requires deepdish')
        data = (nco, co)

        import warnings
        with warnings.catch_warnings():
            dd.io.save(args.save_data_as, data)

else:
    try:
        import deepdish as dd
    except ImportError:
        raise ImportError('saving output requires deepdish')
    nco, co = dd.io.load(args.load_data_from)

# non-candidate regression
nco_reg = cre.Regression(nco)
