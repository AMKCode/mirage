# Copyright 2024 CMU
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import os
from os import path
import sys
import sysconfig
from setuptools import find_packages

# need to use distutils.core for correct placement of cython dll
if "--inplace" in sys.argv:                                                
    from distutils.core import setup
    from distutils.extension import Extension                              
else:
    from setuptools import setup
    from setuptools.extension import Extension\


from distutils.sysconfig import get_config_vars
# Ensure `mpicxx` is used as the C++ compiler
os.environ['CXX'] = '/usr/bin/mpicxx'  # Set the C++ compiler to mpicxx
os.environ['CC'] = '/usr/bin/mpicc'   # Optional: Set the C compiler to mpicc if needed

# Adjust default compiler flags if needed
cfg_vars = get_config_vars()
for key, value in cfg_vars.items():
    if isinstance(value, str) and "gcc" in value:
        cfg_vars[key] = value.replace("gcc", "mpicc").replace("g++", "mpicxx")


def config_cython():
    sys_cflags = sysconfig.get_config_var("CFLAGS")
    try:
        from Cython.Build import cythonize
        ret = []
        cython_path = path.join(path.dirname(__file__), "mirage/_cython")
        mirage_path = path.join(path.dirname(__file__), "..")

        # MPI include and library paths (adapt as needed for your system)
        mpi_include = os.getenv("MPI_INCLUDE", "/usr/include")  # Default MPI include path
        mpi_lib = os.getenv("MPI_LIB", "/usr/lib")  # Default MPI library path
        mpi_libraries = ["mpi", "mpicxx"]  # Name of the MPI library

        with open("outputcython.txt", "w") as file:
            file.write(mpi_include)
            file.write(mpi_lib)

        for fn in os.listdir(cython_path):
            if not fn.endswith(".pyx"):
                continue
            ret.append(Extension(
                "mirage.%s" % fn[:-4],
                ["%s/%s" % (cython_path, fn)],
                include_dirs=[path.join(mirage_path, "include"),
                              path.join(mirage_path, "deps", "json", "include"),
                              path.join(mirage_path, "deps", "cutlass", "include"),
                              "/usr/local/cuda/include",
                              mpi_include],
                libraries=["mirage_runtime", "cudadevrt", "cudart_static", "cudnn", "cublas", "cudart", "cuda", "z3", *mpi_libraries],
                library_dirs=[path.join(mirage_path, "build"),
                              path.join(mirage_path, "deps", "z3", "build"),
                              "/usr/local/cuda/lib",
                              "/usr/local/cuda/lib64",
                              "/usr/local/cuda/lib64/stubs",
                              mpi_lib],
                extra_compile_args=["-std=c++17", "-fpermissive"],
                extra_link_args=["-fPIC"],
                language="c++"))
        return cythonize(ret, compiler_directives={"language_level" : 3})
    except ImportError:
        print("WARNING: cython is not installed!!!")
        raise SystemExit(1)
        return []

setup_args = {}

#if not os.getenv('CONDA_BUILD'):
#    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
#    for i, path in enumerate(LIB_LIST):
#    LIB_LIST[i] = os.path.relpath(path, curr_path)
#    setup_args = {
#        "include_package_data": True,
#        "data_files": [('mirage', LIB_LIST)]
#    }

setup(name='mirage',
      version="0.1.1",
      description="Mirage: A Multi-Level Superoptimizer for Tensor Algebra",
      zip_safe=False,
      install_requires=[],
      packages=find_packages(),
      url='https://github.com/mirage-project/mirage',
      ext_modules=config_cython(),
      #**setup_args,
      )
