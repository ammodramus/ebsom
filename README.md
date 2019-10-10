This repository implements a novel [empirical
Bayesian](https://en.wikipedia.org/wiki/Empirical_Bayes_method) model for
low-frequency variant calling with high-coverage
pooled-sample/high-ploidy/somatic DNA sequencing data. This approach works well
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

For a more concrete mathematical description, see README.ipynb.

This module requires the following Python packages:

 - [h5py](https://www.h5py.org)
 - [tensorflow 1.14+](https://www.tensorflow.org)
 - [pysam](https://github.com/pysam-developers/pysam)

[Cython](https://cython.org) is also required.

Once dependencies are installed, clone this directory and run `python setup.py build_ext -i`.
