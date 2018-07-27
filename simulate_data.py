import h5py
import argparse
import numpy as np
import numpy.random as npr
from scipy.special import logsumexp

import h5py_util
import cyutil as cut
import beta_with_spikes_integrated as bws

parser = argparse.ArgumentParser(
        description='simulate processed read call data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('template', help = 'template HDF5 file')
parser.add_argument('parameters', help = 'file of parameters, one per line')
parser.add_argument('output', help = 'output HDF5 filename')
parser.add_argument('--num-pf-parameters', type = int, default = 3)
parser.add_argument('--num-frequencies', type = int, default = 100)
parser.add_argument('--frequency-output', help = 'file for simulated allele frequencies')
# parser.add_argument('--resample-covariates')
args = parser.parse_args()

fin = h5py.File(args.template)

print '# loading all_majorminor'
all_majorminor = h5py_util.get_major_minor(fin)
print '# obtaining column names'
colnames_str = fin.attrs['covariate_column_names']
colnames = colnames_str.split(',')

h5cm = fin['covariate_matrices']
h5lo = fin['locus_observations']

print '# getting covariate matrix keys'
h5cm_keys = []
for chrom, chrom_cm in h5cm.iteritems():
    for bam, bam_cm in chrom_cm.iteritems():
        print '# getting covariate matrix keys: {}'.format(bam)
        for locus, locus_cm in bam_cm.iteritems():
            spname = locus_cm.name.split('/')
            name = unicode('/'.join(spname[2:]))
            h5cm_keys.append(name)

print '# getting covariate matrix keys'
h5lo_keys = []
for chrom, chrom_lo in h5lo.iteritems():
    for bam, bam_lo in chrom_lo.iteritems():
        print '# getting locus observation keys: {}'.format(bam)
        for locus, locus_lo in bam_lo.iteritems():
            spname = locus_lo.name.split('/')
            name = unicode('/'.join(spname[2:]))
            h5lo_keys.append(name)
assert set(h5lo_keys) == set(h5cm_keys), "covariate matrix and locus observation keys differ"

params = np.loadtxt(args.parameters)

locus_keys = h5cm_keys

regkeys = [(b, r) for b in 'ACGT' for r in (1,2)]
rowlen = int(fin.attrs['rowlen'])
blims = {}
for i, reg in enumerate(regkeys):
    low = rowlen*3*i
    high = rowlen*3*(i+1)
    blims[reg] = (low, high)

nbetas = len(regkeys)*3*rowlen

pf_params = params[-args.num_pf_parameters:]

num_f = args.num_frequencies
freqs = bws.get_freqs(num_f)
windows = bws.get_window_boundaries(num_f)
lf = np.log(freqs)
l1mf = np.log(1-freqs)
logpf = bws.get_lpf(pf_params, freqs, windows)
pf = np.exp(logpf)

if args.frequency_output:
    freq_out = open(args.frequency_output, 'w')

with h5py.File(args.output, 'w') as fout:
    fout_lo = fout.create_group('locus_observations')
    for key_idx, key in enumerate(locus_keys):
        if key_idx % 10 == 0:
            print '# locus {} of {}'.format(key_idx, len(locus_keys))
        X = h5cm[key][:,:]
        lo = h5lo[key]
        spkey = key.split('/')
        bamkey = '/'.join(spkey[:-1])
        locus_idx = int(spkey[-1])
        major, minor = fin['major_minor'][bamkey][locus_idx]
        if major == 'N':
            continue
        if minor == 'N':
            minor_candidates = [el for el in list('ACGT') if el != major]
            minor = npr.choice(minor_candidates, size = 1)[0]

        # get logprobs
        logprobs = {}
        betas = params[:-args.num_pf_parameters]
        rmajor, rminor = cut.comp(str(major)), cut.comp(str(minor))
        for reg in regkeys:
            low, high = blims[reg]
            b = betas[low:high].reshape((rowlen,-1), order = 'F')
            Xb = np.column_stack((np.dot(X,b), np.zeros(X.shape[0])))
            Xb -= logsumexp(Xb, axis = 1)[:,None]
            logprobs[reg] = Xb

        # draw random frequency from allele frequency distribution

        freq = npr.choice(freqs, size = 1, p = pf)

        if args.frequency_output:
            freq_out.write('\t'.join(key.split('/') + [str(freq)]))
        
        major_logprob_key_by_direc = {'f1': (major, 1), 'f2': (major, 2),
                'r1': (rmajor, 1), 'r2': (rmajor, 2)}
        minor_logprob_key_by_direc = {'f1': (minor, 1), 'f2': (minor, 2),
                'r1': (rminor, 1), 'r2': (rminor, 2)}
        directions = ['f1', 'f2', 'r1', 'r2']

        out_locus = fout_lo.create_group(key)

        for direc in directions:
            direc_lo = lo[direc][:]
            direc_obs = []
            for distobs in direc_lo:
                cm_idx = distobs[0]
                nobs = distobs[1:].sum()
                # draw binomial distribution of true allele types
                num_minor = int(npr.binomial(nobs, freq, size = 1)[0])
                num_major = int(nobs-num_minor)
                # draw the observation types
                minor_logprobs = logprobs[minor_logprob_key_by_direc[direc]]
                minor_obs = npr.multinomial(num_minor, np.exp(minor_logprobs[cm_idx]))

                major_logprobs = logprobs[major_logprob_key_by_direc[direc]]
                major_obs = npr.multinomial(num_major, np.exp(major_logprobs[cm_idx]))
                all_obs = minor_obs + major_obs
                tobs = np.concatenate(((cm_idx,), all_obs))
                direc_obs.append(tobs)
            out_locus.create_dataset(direc, data = np.array(direc_obs).astype(np.int), compression = 'gzip',
                    compression_opts = 9)
    fout.copy(fin['covariate_matrices'], 'covariate_matrices')
    fout.copy(fin['major_minor'], 'major_minor')
    for key in fin.attrs.keys():
        fout.attrs[key] = fin.attrs[key]
