import distutils.ccompiler
from distutils.command.build_ext import build_ext
from distutils import sysconfig
from distutils.core import setup, Extension
import numpy as np

conf = sysconfig.get_config_vars()
conf['CFLAGS'] = "-fno-strict-aliasing -fno-common -dynamic -pipe -fno-common -fno-strict-aliasing -fwrapv -mno-fused-madd -Wall -Wstrict-prototypes -Wshorten-64-to-32 -fwrapv -Wall -Wstrict-prototypes -arch i386 -arch x86_64 -O3 -march=core2"
conf['PY_CFLAGS'] = "-fno-strict-aliasing -fno-common -dynamic-pipe -fno-common -fno-strict-aliasing -fwrapv -mno-fused-madd -Wall -Wstrict-prototypes -Wshorten-64-to-32 -fwrapv -Wall -Wstrict-prototypes -I. -IInclude -I./Include -pipe -DPy_BUILD_CORE -arch i386 -arch x86_64 -march=core2 -O3"
conf['OPT'] = "-fwrapv -Os -Wall -Wstrict-prototypes -O3 -march=core2"

# to run:
# python setup.py build
# cp build/<something>/c_module.so .
# python c_module_example.py
module = Extension('hex', sources = ['hex.c'])
setup(name = 'C module example',
      version = '1.0',
      ext_modules = [module],
      include_dirs = [np.get_include()])
