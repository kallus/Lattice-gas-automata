from distutils.core import setup, Extension
import numpy as np

# to run:
# python setup.py build
# cp build/<something>/c_module.so .
# python c_module_example.py
module = Extension('c_module', sources = ['c_module.c'])
setup(name = 'C module example',
      version = '1.0',
      ext_modules = [module],
      include_dirs = [np.get_include()])
