from setuptools import setup
from torch.utils import cpp_extension

setup(name='custom_group_norm',
      ext_modules=[cpp_extension.CppExtension('custom_group_norm', ['main.cpp'],
                                              include_dirs = ['/usr/local/include/eigen3/Eigen'])],
      license='Apache License v2.0',
      cmdclass={'build_ext': cpp_extension.BuildExtension})
      
      
#/home/mitesh/eigen/Eigen
#/usr/local/include/eigen3/Eigen
