from distutils.core import Extension,setup
from Cython.Build import cythonize

ext = Extension(name="pixel_iteration", sources=["pixel_iteration.pyx"],compiler_directives={'language_level' : "3"})
setup(ext_modules = cythonize(ext))