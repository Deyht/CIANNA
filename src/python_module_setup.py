from distutils.core import setup, Extension
setup(name = 'CIANNA', version = '1.0', ext_modules = [Extension('CIANNA', ['python_module.c'], extra_objects=['conv_layer.o', 'dense_layer.o', 'pool_layer.o', 'activ_functions.o', 'initializers.o', 'vars.o', 'auxil.o', 'cuda_main.o', 'cuda_conv_layer.o', 'cuda_dense_layer.o', 'cuda_pool_layer.o', 'cuda_activ_functions.o'], extra_link_args=['-L/usr/local/cuda-10.1/lib64', '-lcudart', '-lcublas', '-std=c99'], define_macros=[('CUDA','1')])])



