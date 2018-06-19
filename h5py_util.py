import h5py
import numpy as np
from multiprocessing.sharedctypes import RawArray
import ctypes


def add_major_minor(all_majorminor, h5mm):
    for chrom, chrom_mm in all_majorminor.iteritems():
        h5chrom = h5mm.create_group(chrom)
        for bam, bam_mm in chrom_mm.iteritems():
            bam_dat = np.array(bam_mm, dtype = 'S1')
            h5chrom.create_dataset(bam, dtype = 'S1', shape = bam_dat.shape, data = bam_dat)

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


empty_locobs = lambda: (
        (np.empty((0, 5), dtype = np.uint32), np.empty((0, 5), dtype = np.uint32)),
        (np.empty((0, 5), dtype = np.uint32), np.empty((0, 5), dtype = np.uint32)))


'''
def get_locus_locobs(h5lo_locus):
    f1shape = h5lo_locus['f1'].shape
    f2shape = h5lo_locus['f2'].shape
    r1shape = h5lo_locus['r1'].shape
    r2shape = h5lo_locus['r2'].shape

    loc_obs = ((
                get_raw_uint32_array(np.prod(f1shape)),
                get_raw_uint32_array(np.prod(f2shape))
            ),
            (
                get_raw_uint32_array(np.prod(r1shape)),
                get_raw_uint32_array(np.prod(r2shape))
            ))

    x = memoryview(loc_obs[0][0])
    y = h5lo_locus['f1'][:,:].reshape(np.prod(f1shape))
    x[:] = y
    loc_obs[0][0].shape = f1shape

    x = memoryview(loc_obs[0][1])
    y = h5lo_locus['f2'][:,:].reshape(np.prod(f2shape))
    x[:] = y
    loc_obs[0][1].shape = f2shape

    x = memoryview(loc_obs[1][0])
    y = h5lo_locus['r1'][:,:].reshape(np.prod(r1shape))
    x[:] = y
    loc_obs[1][0].shape = r1shape

    x = memoryview(loc_obs[1][1])
    y = h5lo_locus['r2'][:,:].reshape(np.prod(r2shape))
    x[:] = y
    loc_obs[1][1].shape = r2shape

    return loc_obs
'''

def get_locus_locobs(h5lo_locus):
    f1shape = h5lo_locus['f1'].shape
    f2shape = h5lo_locus['f2'].shape
    r1shape = h5lo_locus['r1'].shape
    r2shape = h5lo_locus['r2'].shape

    loc_obs = ((
                h5lo_locus['f1'][:,:],
                h5lo_locus['f2'][:,:]
            ),
            (
                h5lo_locus['r1'][:,:],
                h5lo_locus['r2'][:,:]
            ))

    return loc_obs

'''
def get_locus_locobs(h5lo_locus):
    f1shape = h5lo_locus['f1'].shape
    f2shape = h5lo_locus['f2'].shape
    r1shape = h5lo_locus['r1'].shape
    r2shape = h5lo_locus['r2'].shape

    loc_obs = ((
                h5lo_locus['f1'],
                h5lo_locus['f2']
            ),
            (
                h5lo_locus['r1'],
                h5lo_locus['r2']
            ))

    return loc_obs
'''


def get_raw_uint32_array(shape):
    n_el = np.prod(shape)
    arr = np.frombuffer(RawArray(ctypes.c_uint32, n_el), dtype = np.uint32).reshape(shape)
    return arr

def get_locobs(h5in, mm, update_interval = 100):
    lo = {}
    for chrom, h5lo_chrom in h5in['locus_observations'].iteritems():
        chrom_mm = mm[chrom]
        lo[chrom] = {}
        bams = h5lo_chrom.keys()
        for bam in bams:
            h5lo_bam = h5lo_chrom[bam]
            print '# loading locus observations for {} ({} of {})'.format(bam, bams.index(bam)+1, len(bams))
            bam_mm = chrom_mm[bam]
            seqlen = bam_mm.shape[0]
            bam_lo = []
            #for i in xrange(1, seqlen+1):
            for i in xrange(0, seqlen):
                loc_str = str(i)
                if loc_str in h5lo_bam:
                    loc_obs = get_locus_locobs(h5lo_bam[loc_str])
                    bam_lo.append(loc_obs)
                else:
                    bam_lo.append(empty_locobs())
            assert len(bam_lo) == seqlen
        lo[chrom][bam] = bam_lo
    return lo
