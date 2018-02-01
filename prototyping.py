from __future__ import division, print_function
import argparse
import numpy as np
import pysam

import util as ut

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
args = parser.parse_args()

bam_fns, bams = ut.get_bams(args.bams)
ref_names = get_ref_names(args.references, bams)
all_counts = ut.get_all_all_counts(bams, ref_names, args.min_bq)
all_freqs = ut.get_freqs(all_counts)
is_candidate = ut.determine_candidates(all_freqs, args.min_candidate_freq)
all_consensuses = ut.get_all_consensuses(all_freqs, min_coverage = 20)

# make rowmakers
row_makers, rowlen = ut.get_row_makers(bam_fns, ref_names, args.context_length, 
        args.round_distance_by, all_consensuses)
nc_observations = NonCandidateCollector(rowlen)
c_observations = CandidateCollector(rowlen)
            
for ref in ref_names:
    for bam_fn in bam_fns:
        bam = bams[bam_fn]
        counts = all_counts[ref][bam_fn]
        freqs = all_freqs[ref][bam_fn]
        is_can = is_candidate[ref][bam_fn]

        
for entry in aln.header['SQ']:
    ref = entry['SN']
    rlen = entry['LN']
    i = 0
    if ref != 'chrM':
        continue
    for read in aln.fetch(contig = ref, start = 0, end = rlen):
        mapq = read.mapping_quality
        if mapq < min_mq:
            continue
        seq = read.seq
        readlen = read.alen  # aligned length
        qualities = read.query_qualities
        reverse = read.is_reverse
        readnum = read.is_read2 + 1
        for qpos, refpos in read.get_aligned_pairs(True):
            if qpos is None or refpos is None:
                continue
            q = qualities[qpos]
            if q < min_bq:
                continue
            if reverse:
                if qpos >= readlen-context_len:
                    continue
                context = rev_comp(seq[qpos+1:qpos+3])
                if 'N' in context:
                    continue
                obsbase = rev_comp(seq[qpos])
                consbase = rev_comp(consensus[refpos])
                dend = readlen - qpos
            else:
                if qpos < context_len:
                    continue
                context = seq[qpos-context_len:qpos]
                if 'N' in context:
                    continue
                obsbase = seq[qpos]
                consbase = consensus[refpos]
                dend = qpos
            if obsbase == 'N' or consbase == 'N':
                continue
            row = row_makers[consbase].get_covariate_row(consbase, q, mapq, context, dend, refpos, aln_fn, reverse)
            if not is_candidate[refpos]:
                nc_observations[(consbase, readnum, obsbase)].add(row)
            else:
                c_observations.add(row, refpos, consbase, readnum, obsbase)
                
            #if consbase != obsbase:
            #    print(consbase, obsbase, q, qpos, refpos)
        # base quality, map quality, distance from beginning, contaminant frequency, sequence context dummy var, constant
        # plus, user-defined
        i += 1
        if i >= 10000:
            break
