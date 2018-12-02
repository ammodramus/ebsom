from __future__ import division, print_function
import argparse
import numpy as np
import pysam
from locuscollectors import NonCandidateCollector, CandidateCollector
import util as ut
import cyprocessreads as cpr
import cyregression as cre
import cygetcolumndata as cgcd
import h5py
import h5py_util

import beta_with_spikes_integrated as bws
import retry_tensorflow_neural_net as rt

import numpy.random as npr

# just for testing
import deepdish as dd

import time
import mpi4py
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()


def complement(bases):
    return ''.join(['TGCA'['ACGT'.index(base)] for base in bases])

def get_context_data(
        cons,
        position,
        context_len,
        circular=True,
        onehot=False,
        max_context_size=100):
    # note: position is already 0-based
    conslen = len(cons)
    if position < 0 or position >= conslen:
        raise ValueError('cannot get context for position {}: invalid position'.format(
            position+1))

    # check it's valid if the chromosome isn't circular
    if not circular:
        if position-context_len<0 or position+context_len >= conslen:
            raise ValueError("cannot get context for position {} "
                             "with reference length {}".format(position,conslen))

    if circular:
        cons = cons + cons + cons  # 1) add the same sequence before and after
        position += conslen        # 2) amend position to reflect new sequence (note conslen not used again)

    forward_context_bases = cons[position-context_len:position]
    #reverse_context_bases = cgcd.complement(cons[position:position-context_len:-1])   # still forward translation
    reverse_context_bases = complement(cons[position+2:position:-1])   # still forward translation
    if onehot:
        raise NotImplementedError('onehot not yet implemented')
    else:
        forward_context = np.zeros(4*context_len, dtype=np.float32)
        for i in xrange(context_len):
            forward_context[i*4+'ACGT'.index(forward_context_bases[i])] = 1
        reverse_context = np.zeros(4*context_len, dtype=np.float32)
        for i in xrange(context_len):
            reverse_context[i*4+'ACGT'.index(reverse_context_bases[i])] = 1

    return forward_context, reverse_context



def get_contamination(position_consensuses):
    forward_bases, forward_counts = np.unique(position_consensuses,
                                              return_counts=True)
    forward_fracs = forward_counts.astype(np.float64)/len(position_consensuses)
    forward_contam = np.zeros(4, dtype=np.float32)
    for base, frac in zip(forward_bases, forward_fracs):
        forward_contam['ACGT'.index(base)] = frac
    reverse_contam = forward_contam[::-1].copy()
    return forward_contam, reverse_contam


def get_bam_data(bam_fn, all_bam_fns):
    bam_data = np.zeros(len(all_bam_fns), dtype=np.float32)
    bam_data[all_bam_fns.index(bam_fn)] = 1.0
    return bam_data




desc = 'jointly infer sequencing error profiles and polymorphisms'
parser = argparse.ArgumentParser(
        description=desc,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('bams', help='file containing list of bam files')
parser.add_argument('references',
        help = 'name of file containing list of reference sequence names, or '
               'comma-separated list of reference sequence names')
parser.add_argument('--min-bq', help='minimum base quality',
        type = ut.positive_int, default=20)
parser.add_argument('--min-mq', help='minimum mapping quality',
        type=ut.positive_int, default=20)
parser.add_argument('--context-length', type=ut.nonneg_int, default=2,
        help='number of preceding bases to use as covariate')
parser.add_argument('--round-distance-by',
        type=ut.positive_int, default=1,
        help='round distance from start of read by this amount. larger '
             'numbers make for more compression in the data, faster '
             'likelihood evaluations.')
parser.add_argument('--min-coverage', type=int, default=20,
                    help="minimum coverage for getting consensus")
args = parser.parse_args()
min_bq = args.min_bq
min_mq = args.min_mq
context_len = args.context_length

# break bam names into prefix/name
prefix, bam_fns, bams = ut.get_bams(args.bams)
ref_names = ut.get_ref_names(args.references, bams)

num_bams = len(bam_fns)

all_counts = None
all_consensuses = None
all_majorminor = None
if rank == 0:
    print('getting counts')
    all_counts = ut.get_all_counts(bams, ref_names, min_bq)
    print('getting consensus')
    all_consensuses = ut.get_all_consensuses(all_counts, min_coverage=args.min_coverage)
    print('getting major-minor')
    all_majorminor = ut.get_all_majorminor(all_counts)
else:
    all_counts = None
    all_consensuses = None
    all_majorminor = None
all_counts = comm.bcast(all_counts, root=0)
all_consensuses = comm.bcast(all_consensuses, root=0)
all_majorminor = comm.bcast(all_majorminor, root=0)

#savedat = (all_counts, all_consensuses, all_majorminor)
#dd.io.save('debug_data.h5', savedat)

##################################################################################
# for debugging only
###################################
#print('(loading data from deepdish for prototyping purposes)')
#all_counts, all_consensuses, all_majorminor = dd.io.load('debug_data.h5')

##################################################################################

all_loci = []
for ref in ref_names:
    for bamfn in bams.keys():
        cons = all_consensuses[ref][bamfn]
        reflen = len(cons)
        for i in range(1,reflen+1):
            all_loci.append((ref, bamfn, i))
all_loci = np.array(all_loci)


def get_data(dataslice):
    ref, bamfn, position = dataslice
    bam = bams[bamfn]
    ref = bytes(ref)
    position_one = int(position)   # 0-based
    position_zero = position_one-1
    try:
        position_consensuses = map(
            lambda bamfn: all_consensuses[ref][bamfn][position_zero],
            all_consensuses[ref].keys())
    except:
        print('bad position:', position_one)
    cons = all_consensuses[ref][bamfn]
    major, minor = all_majorminor[ref][bamfn][position_zero]
    reflen = len(cons)
    start = time.time()
    forward_data, reverse_data = cgcd.get_column_data(
        bam, ref, reflen, position_zero, args.min_bq, args.min_mq,
        args.context_length, bytes(cons), args.round_distance_by)
    dur = time.time()-start
    for_cov = forward_data.covariate_matrix().copy().astype(np.float32)
    for_obs = forward_data.observations().copy().astype(np.float32)
    rev_cov = reverse_data.covariate_matrix().copy().astype(np.float32)
    rev_obs = reverse_data.observations().copy().astype(np.float32)

    forward_context, reverse_context = get_context_data(cons, position_zero,
            args.context_length)
    forward_contam, reverse_contam = get_contamination(position_consensuses)

    bam_data = get_bam_data(bamfn, bam_fns)

    forward_const_cov = np.concatenate((forward_context, forward_contam, bam_data))
    reverse_const_cov = np.concatenate((reverse_context, reverse_contam, bam_data))
    return (for_cov, for_obs, rev_cov, rev_obs, forward_const_cov,
            reverse_const_cov, major, minor, position_one)


size = comm.Get_size()

if rank == 0:
    '''
    import h5py
    fout = h5py.File('test.h5', 'w')
    data_group = fout.create_group('data')
    indices_group = fout.create_group('meta')
    # note that with gzip, compression level only slows down compression, not
    # decompression
    for_cov_data = data_group.create_dataset('for_cov', shape=(100000, 4),
            dtype=np.float32, maxshape=(None,4),compression="gzip", compression_opts=9,
            chunksize=(2000,4))
    rev_cov_data = data_group.create_dataset('rev_cov', shape=(100000, 4),
            dtype=np.float32, maxshape=(None,4),compression="gzip", compression_opts=9,
            chunksize=(2000,4))
    for_obs_data = data_group.create_dataset('for_obs', shape=(100000, 4),
            dtype=np.uint16, maxshape=(None,4),compression="gzip", compression_opts=9,
            chunksize=(2000,4))
    rev_obs_data = data_group.create_dataset('rev_obs', shape=(100000, 4),
            dtype=np.uint16, maxshape=(None,4),compression="gzip", compression_opts=9,
            chunksize=(2000,4))
    '''
    import tables
    h5file = tables.File('test.h5', 'w', title='test file')
    comp = tables.Filters(complevel=9, complib='blosc')
    data_group = h5file.create_group(h5file.root, 'data')
    meta_group = h5file.create_group(h5file.root, 'meta')

    # read data
    for_cov_data = tables.EArray(h5file.root.data, 'for_cov',
                                 tables.Atom.from_dtype(np.dtype(np.float32)),
                                 shape=(0,4), filters=comp)
    rev_cov_data = tables.EArray(h5file.root.data, 'rev_cov',
                                 tables.Atom.from_dtype(np.dtype(np.float32)),
                                 shape=(0,4), filters=comp)
    for_obs_data = tables.EArray(h5file.root.data, 'for_obs',
                                 tables.Atom.from_dtype(np.dtype(np.uint16)),
                                 shape=(0,4), filters=comp)
    rev_obs_data = tables.EArray(h5file.root.data, 'rev_obs',
                                 tables.Atom.from_dtype(np.dtype(np.uint16)),
                                 shape=(0,4), filters=comp)

    #               context            contam.       bam
    ncols_const = 4*args.context_length + 4 + len(bam_fns)

    # meta (position) data
    '''
    each position has:
        - identifier information:
            - reference name (Enum)
            - bam name (Enum)
            - position (uint64)
            - unique identifier (uint64)
        - major
        - minor
        - forward start index (for for_cov and for_obs, uint64)
        - forward end index (for for_cov and for_obs, uint64)
        - reverse start index (for rev_cov and rev_obs, uint64)
        - reverse end index (for rev_cov and rev_obs, uint64)
        - site covariates (float32, each)
    '''

    ref_names_enum = tables.Enum(ref_names)
    bam_fns_enum = tables.Enum(bam_fns)

    class SiteMetadata(tables.IsDescription):
        reference = tables.EnumCol(ref_names_enum, ref_names[0], base='uint8')
        bam = tables.EnumCol(bam_fns_enum, bam_fns[0], base='uint8')
        position = tables.UInt64Col()
        idnumber = tables.UInt64Col()
        major = tables.StringCol(1)
        minor = tables.StringCol(1)
        forward_start = tables.UInt64Col()
        forward_end = tables.UInt64Col()
        reverse_start = tables.UInt64Col()
        reverse_end = tables.UInt64Col()
        forward_const_cov = tables.Float32Col(shape=(ncols_const,))
        reverse_const_cov = tables.Float32Col(shape=(ncols_const,))

    meta_table = h5file.create_table(meta_group, 'metadata', SiteMetadata,
                                     'Per-site metadata')
    meta_row = meta_table.row


    # each position has forward start position, forward end position (non-inclusive), 




cur_id_number = 0
chunk_size = 200
for i in range(0, all_loci.shape[0], chunk_size):
    chunk = all_loci[i:i+chunk_size,:]
    split_chunk = np.array_split(chunk, size)
    slicedata = comm.scatter(split_chunk, root=0)

    proc_data = []
    for sl in slicedata:
        proc_data.append(get_data(sl))

    chunk_data = comm.gather(proc_data, root=0)
    if rank == 0:
        chunk_data = reduce(lambda x,y: list(x)+list(y), chunk_data)
        for chunkp, chunk_dat in zip(chunk, chunk_data):
            (for_cov, for_obs, rev_cov, rev_obs, forward_const_cov,
             reverse_const_cov, major, minor, position_one) = chunk_dat

            ref, bamfn, position = chunkp
            meta_row['reference'] = ref_names_enum[ref]
            meta_row['bam'] = bam_fns_enum[bamfn]
            meta_row['position'] = position_one
            meta_row['idnumber'] = cur_id_number; cur_id_number += 1
            meta_row['major'] = major[0]
            meta_row['minor'] = minor[0]
            meta_row['forward_start'] = for_cov_data.shape[0]
            meta_row['forward_end'] = for_cov_data.shape[0] + for_cov.shape[0]
            meta_row['reverse_start'] = rev_cov_data.shape[0]
            meta_row['reverse_end'] = rev_cov_data.shape[0] + rev_cov.shape[0]
            meta_row['forward_const_cov'] = forward_const_cov
            meta_row['reverse_const_cov'] = reverse_const_cov
            meta_row.append()

            assert for_cov.shape[0] == for_obs.shape[0]
            assert rev_cov.shape[0] == rev_obs.shape[0]

            for_cov_data.append(for_cov)
            rev_cov_data.append(rev_cov)
            for_obs_data.append(for_obs)
            rev_obs_data.append(rev_obs)
        print(chunk[0])
