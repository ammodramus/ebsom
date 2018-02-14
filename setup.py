from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension("cylocobs", ["cylocobs.pyx"],
        include_dirs = [np.get_include()],
        libraries = [],
        library_dirs = []),
    Extension("cyprocessreads", ["cyprocessreads.pyx"],
        include_dirs = [np.get_include()],
        libraries = [],
        library_dirs = []),
    Extension("cyrowmaker", ["cyrowmaker.pyx"],
        include_dirs = [np.get_include()],
        libraries = [],
        library_dirs = []),
    Extension("cyregression", ["cyregression.pyx"],
        include_dirs = [np.get_include()],
        libraries = [],
        library_dirs = []),
    Extension("cyregcov", ["cyregcov.pyx"],
        include_dirs = [np.get_include()],
        libraries = [],
        library_dirs = []),
    Extension("cylocll", ["cylocll.pyx"],
        include_dirs = [np.get_include()])
]
setup(
    ext_modules = cythonize(extensions),
)
