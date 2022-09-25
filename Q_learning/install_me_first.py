"""
Install the cython first as follows:

python3 install_me_first.py build_ext --inplace    

"""
#from distutils.core import setup
#from Cython.Build import cythonize
#setup(ext_modules=cythonize('evaluations.pyx'))


from setuptools import setup, Extension

module = Extension('example', sources=['evaluations.pyx'])

setup(
    name='cythonTest',
    version='1.0',
    author='jetbrains',
    ext_modules=[module]
)