#cython: boundscheck=False, wraparound=False, nonecheck=False
from distutils.core import Extension,setup
from Cython.Build import cythonize

ext = Extension(name="golomb_cython", sources=["golomb_cython.pyx"],compiler_directives={'language_level' : "3",'boundscheck':False, 'wraparound':False, 'nonecheck':False},libraries=["m"])
setup(ext_modules = cythonize(ext))