This repository implements a novel [empirical
Bayesian](https://en.wikipedia.org/wiki/Empirical_Bayes_method) model for
low-frequency SNP calling with high-coverage
pooled-sample/high-ploidy/somatic DNA sequencing data. This prototype works well
on simulated data but is **not currently working on the real data for which it
was designed. Use with caution.**

The basic approach is to jointly model the DNA sequencing error process and the
global allele-frequency distribution, obtaining a maximum-likelihood estimate
of each. The inferred global distribution of allele frequencies is employed as
the prior distribution in calculating the posterior distribution of allele
frequencies at individual candidate loci, making this an empirical Bayesian
approach.

The sequencing error model is a fully-connected artificial neural network,
where for each base-call the inputs are a number of covariates associated with
that base-call, including the true base, base quality, mapping quality, read
number, position along the read, and potential for contamination at the site
(allowing for index-swapping of multiplexed libraries).

For a more concrete (but still brief) mathematical description, see [README.pdf](https://github.com/ammodramus/ebsom/blob/master/README.pdf).

## Installation

This module requires the following Python packages:

 - [h5py](https://www.h5py.org)
 - [tensorflow 2.0+](https://www.tensorflow.org)
 - [pysam](https://github.com/pysam-developers/pysam)

[Cython](https://cython.org) is also required.

Once dependencies are installed, clone this directory and run `python setup.py
build_ext -i`.

## Usage

There are three steps to running this model and calculating the posterior
distribution of allele frequencies.

 1. Collect the covariate data from the .BAM alignments:
       ```
       python collect_data.py bamlist.txt chromlist.txt data.h5
       ```
    Here `bamlist.txt` is a file containing a list of .BAM alignments from the
    same sequencing lane, `chromlist.txt` is a file containing a list of
    chromosomes/contigs to analyze (or the name of a single chromosome/contig),
    and `output.h5` is the name of the HDF5 data file to be created. See
    `python collect_data.py -h` for details and more options.
 2. Optimize the global model:
       ```
       python run_model.py data.h5 --save-model error.model
       ```
    `data.h5` is the data file produced in the first step, and `error.model` is
    the filepath (a directory, on POSIX systems) in which the optimized model
    parameters will be saved. See `python run_model.py -h` for options that can
    be used to control the optimization.
 3. Calculate and output the (log-) posterior distribution of allele
    frequencies at each site:
       ```
       python calculate_posterior_probabilities.py data.h5 error.model
       ```
    `data.h5` and `error.model` are as in the previous two steps. This prints
    to STDOUT a tab-separated table of the most-likely frequency (i.e., the MAP
    estimate) and the entire log-posterior distribution of the discrete allele
    frequencies for each alignment file, chromosome, and position. The discrete
    allele frequencies are output in a '#'-prefixed comment line.
