"""
Setup script for CUDA Time Series Python package
"""

from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir
from setuptools import setup, Extension
import pybind11
import os
import sys


cuda_home = os.environ.get('CUDA_HOME', '/usr/local/cuda')
if not os.path.exists(cuda_home):
    raise RuntimeError(f"CUDA_HOME not found at {cuda_home}. Please set CUDA_HOME environment variable.")

cuda_include = os.path.join(cuda_home, 'include')
cuda_lib = os.path.join(cuda_home, 'lib64')

if not os.path.exists(cuda_include):
    raise RuntimeError(f"CUDA include directory not found at {cuda_include}")

# extension module 
ext_modules = [
    Pybind11Extension(
        "cuda_ts_py",
        [
            "src/python/bindings.cpp",
        ],
        include_dirs=[
            "include",
            "src",
            cuda_include,
            pybind11.get_include(),
        ],
        libraries=['cudart', 'cufft', 'cublas'],
        library_dirs=[cuda_lib],
        language='c++',
        cxx_std=17,
        extra_compile_args=['-O3', '-Wall', '-Wextra'],
    ),
]

setup(
    name="cuda-time-series",
    version="0.1.0",
    author="Sindhu Satish",
    description="GPU-accelerated time series analysis library",
    long_description="",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "pandas>=1.2.0",
    ],
    classifiers=[],
)

