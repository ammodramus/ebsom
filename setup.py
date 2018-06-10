from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

profile_macros = []
#profile_macros = [('CYTHON_TRACE', 1)]

extensions = [
    Extension("cylocobs", ["cylocobs.pyx"],
        include_dirs = [np.get_include()],
        libraries = [],
        library_dirs = []),
    Extension("cyprocessreads", ["cyprocessreads.pyx"],
        include_dirs = [np.get_include()],
        libraries = [],
        library_dirs = [],
        define_macros = profile_macros),
    Extension("cyrowmaker", ["cyrowmaker.pyx"],
        include_dirs = [np.get_include()],
        libraries = [],
        library_dirs = [],
        define_macros = profile_macros),
    Extension("cyregression", ["cyregression.pyx"],
        include_dirs = [np.get_include()],
        libraries = [],
        library_dirs = [],
        define_macros = profile_macros),
    Extension("cyutil", ["cyutil.pyx"],
        include_dirs = [np.get_include()],
        libraries = [],
        library_dirs = [],
        define_macros = profile_macros),
    Extension("cyregcov", ["cyregcov.pyx"],
        include_dirs = [np.get_include()],
        libraries = [],
        library_dirs = [],
        define_macros = profile_macros),
    Extension("cylocll", ["cylocll.pyx"],
        include_dirs = [np.get_include()]),
    Extension("cygradient", ["cygradient.pyx"],
        include_dirs = [np.get_include()],
        extra_compile_args = ['-march=native'],
        define_macros = profile_macros),
        #include_dirs = [np.get_include()],
        #define_macros=[('CYTHON_TRACE',1)]),
    Extension("cylikelihood", ["cylikelihood.pyx"],
        include_dirs = [np.get_include()],
        extra_compile_args = ['-march=native'],
        define_macros = profile_macros),
    Extension("doublevec", ["doublevec.pyx"],
        define_macros = profile_macros),
    Extension("doubleveccounts", ["doubleveccounts.pyx"],
        define_macros = profile_macros),
    #Extension("cyglobal", ["cyglobal.pyx"], define_macros=[('CYTHON_TRACE',1)])
    Extension("cyglobal", ["cyglobal.pyx"], include_dirs = [np.get_include()],
        define_macros = profile_macros)
]
setup(
    ext_modules = cythonize(extensions),
)
