import h5py
import numpy as np

def add_major_minor(all_majorminor, h5mm):
    for chrom, chrom_mm in all_majorminor.iteritems():
        h5chrom = h5mm.create_group(chrom)
        for bam, bam_mm in chrom_mm.iteritems():
            bam_dat = np.array(bam_mm, dtype = 'S1')
            h5chrom.create_dataset(bam, dtype = 'S1', shape = bam_dat.shape, data = bam_dat)

def get_major_minor(h5in):
    mm = {}
    for chrom, h5_chrom_mm in h5in['major_minor'].iteritems():
        mm[chrom] = {}
        for bam, h5_bam_mm in h5_chrom_mm.iteritems():
            h5_bam_mm = h5_bam_mm[:,:]
            mm[chrom][bam] = h5_bam_mm
    return mm


empty_locobs = lambda: (
        (np.empty((0, 5), dtype = np.uint32), np.empty((0, 5), dtype = np.uint32)),
        (np.empty((0, 5), dtype = np.uint32), np.empty((0, 5), dtype = np.uint32)))


'''
def get_locobs(h5in, mm):
    lo = {}
    for chrom, h5lo_chrom in h5in['locus_observations'].iteritems():
        chrom_mm = mm[chrom]
        lo[chrom] = {}
        for bam, h5lo_bam in h5lo_chrom.iteritems():
            bam_mm = chrom_mm[bam]
            seqlen = bam_mm.shape[0]
            bam_lo = []
            for i in xrange(1, seqlen+1):
                loc_str = str(i)
                if loc_str in h5lo_bam.keys():
                    loc_obs = ((
                                h5lo_bam[loc_str]['f1'][:,:],
                                h5lo_bam[loc_str]['f2'][:,:],
                            ),
                            (
                                h5lo_bam[loc_str]['r1'][:,:],
                                h5lo_bam[loc_str]['r2'][:,:],
                            ))
                    bam_lo.append(loc_obs)
                else:
                    bam_lo.append(empty_locobs())
            assert len(bam_lo) == seqlen
        lo[chrom][bam] = bam_lo
    return lo
'''

from multiprocessing.sharedctypes import RawArray
import ctypes

def get_raw_uint32_array(shape):
    n_el = np.prod(shape)
    arr = np.frombuffer(RawArray(ctypes.c_uint32, n_el), dtype = np.uint32).reshape(shape)
    return arr

def get_locobs(h5in, mm):
    lo = {}
    for chrom, h5lo_chrom in h5in['locus_observations'].iteritems():
        chrom_mm = mm[chrom]
        lo[chrom] = {}
        for bam, h5lo_bam in h5lo_chrom.iteritems():
            bam_mm = chrom_mm[bam]
            seqlen = bam_mm.shape[0]
            bam_lo = []
            for i in xrange(1, seqlen+1):
                loc_str = str(i)
                if loc_str in h5lo_bam.keys():
                    f1shape = h5lo_bam[loc_str]['f1'].shape
                    f2shape = h5lo_bam[loc_str]['f2'].shape
                    r1shape = h5lo_bam[loc_str]['r1'].shape
                    r2shape = h5lo_bam[loc_str]['r2'].shape

                    loc_obs = ((
                                get_raw_uint32_array(f1shape),
                                get_raw_uint32_array(f2shape)
                            ),
                            (
                                get_raw_uint32_array(r1shape),
                                get_raw_uint32_array(r2shape)
                            ))

                    loc_obs[0][0][:,:] = h5lo_bam[loc_str]['f1'][:,:]
                    loc_obs[0][1][:,:] = h5lo_bam[loc_str]['f2'][:,:]
                    loc_obs[1][0][:,:] = h5lo_bam[loc_str]['r1'][:,:]
                    loc_obs[1][1][:,:] = h5lo_bam[loc_str]['r2'][:,:]

                    bam_lo.append(loc_obs)
                else:
                    bam_lo.append(empty_locobs())
            assert len(bam_lo) == seqlen
        lo[chrom][bam] = bam_lo
    return lo
