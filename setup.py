from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

profile_macros = []
#profile_macros = [('CYTHON_TRACE', 1)]

extensions = [
    Extension("cylocobs", ["cylocobs.pyx"],
        include_dirs=[np.get_include()],
        libraries=[],
        library_dirs=[]),
    Extension("cyprocessreads", ["cyprocessreads.pyx"],
        include_dirs=[np.get_include()],
        libraries=[],
        library_dirs=[],
        define_macros=profile_macros),
    Extension("cyrowmaker", ["cyrowmaker.pyx"],
        include_dirs=[np.get_include()],
        libraries=[],
        library_dirs=[],
        define_macros=profile_macros),
    Extension("cyutil", ["cyutil.pyx"],
        include_dirs=[np.get_include()],
        libraries=[],
        library_dirs=[],
        define_macros=profile_macros),
    Extension("cyregcov", ["cyregcov.pyx"],
        include_dirs=[np.get_include()],
        libraries=[],
        library_dirs=[],
        define_macros=profile_macros)
]

setup(
    ext_modules=cythonize(extensions),
)
