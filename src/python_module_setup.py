
#from distutils.core import setup, Extension
from setuptools import setup, Extension
import numpy
import os


#os.environ['USE_CUDA'] = '1'
#os.environ['USE_BLAS'] = '1'
#os.environ['USE_OPENMP'] = '1'

cuda_obj = []
cuda_extra = []
cuda_include = []
cuda_macro = [(None, None)]

blas_obj = []
blas_extra = []
blas_include = []
blas_macro = [(None, None)]

open_mp_extra = []

if(os.environ.get('USE_CUDA') != None):
	print("USE_CUDA")
	cuda_obj = ['cuda/cuda_main.o', 'cuda/cuda_conv_layer.o', 'cuda/cuda_dense_layer.o', 'cuda/cuda_pool_layer.o', 'cuda/cuda_norm_layer.o', 'cuda/cuda_lrn_layer.o', 'cuda/cuda_activ_functions.o']
	cuda_include = ['/usr/local/cuda-12.0/include']
	cuda_extra = ['-L/usr/local/cuda-12.0/lib64', '-lcudart', '-lcublas', '-lcurand']
	cuda_macro = [('CUDA','1'), ('CUDA_THREADS_PER_BLOCKS', '256')]
if(os.environ.get('USE_BLAS') != None):
	print("USE_BLAS")
	blas_obj = ['blas/blas_dense_layer.o', 'blas/blas_conv_layer.o']
	blas_include = ['/opt/OpenBLAS/include']
	blas_extra = ['-lopenblas', '-L/opt/OpenBLAS/lib']
	blas_macro = [('BLAS', '1')]
if(os.environ.get('USE_OPENMP') != None):
	print("USE_OPENMP")
	open_mp_extra = ['-fopenmp']

setup(name = 'CIANNA', 
	version = '1.0.0.0', 
	ext_modules = [Extension('CIANNA', ['python_module.c'], 
	extra_objects=['conv_layer.o', 'dense_layer.o', 'pool_layer.o', 'norm_layer.o', 'lrn_layer.o', 'activ_functions.o', 'initializers.o', 'vars.o', 'auxil.o', 'naiv/naiv_dense_layer.o', 'naiv/naiv_conv_layer.o', 'naiv/naiv_pool_layer.o', 'naiv/naiv_norm_layer.o'] + cuda_obj + blas_obj,
	include_dirs= cuda_include + blas_include + [numpy.get_include()],
	extra_link_args=['-O3 -std=c99 --disable-gil'] + cuda_extra + blas_extra + open_mp_extra,
	define_macros=[('MAX_LAYERS_NB', '200'), ('MAX_NETWORKS_NB','10')] + cuda_macro + blas_macro)])

