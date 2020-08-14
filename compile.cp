

#	Copyright (C) 2020 David Cornu
#	for the Convolutional Interactive Artificial 
#	Neural Network by/for Astrophysicists (CIANNA) Code
#	(https://github.com/Deyht/CIANNA)
#
#	Licensed under the Apache License, Version 2.0 (the "License");
#	you may not use this file except in compliance with the License.
#	You may obtain a copy of the License at
#
#		http://www.apache.org/licenses/LICENSE-2.0
#
#	Unless required by applicable law or agreed to in writing, software
#	distributed under the License is distributed on an "AS IS" BASIS,
#	WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#	See the License for the specific language governing permissions and
#	limitations under the License.







######################################################
#            COMPILER DEFINES STRUCTURES             #
######################################################

defines_variables="-D MAX_LAYERS_NB=100 -D MAX_NETWOKRS_NB=10 -D CUDA_THREADS_PER_BLOCKS=64"

######################################################
#                  LIBRARY LOCATIONS                 #
######################################################

gcc_compile_dir="/usr/bin/gcc"
openblas_include_dir="/opt/OpenBLAS/include/"
openblas_lib_dir="/opt/OpenBLAS/lib"
cuda_lib_path="/usr/local/cuda-10.2/lib64"
compile_opt="-O3 -fPIC -Wall -Werror -Wno-unused-result -fmax-errors=2 -fbounds-check -Wno-unknown-pragmas"

######################################################

	
for i in $*
do
	if [ $i  = "CUDA" ]
	then
		cuda_arg="$cuda_arg -D CUDA -D comp_CUDA -lcublas -lcudart -arch=sm_60"
		arg="$arg -D CUDA -lcublas -lcudart -L $cuda_lib_path "
		cuda_src="cuda_main.cu cuda_conv_layer.cu cuda_dense_layer.cu cuda_pool_layer.cu cuda_activ_functions.cu"
		cuda_obj="cuda/cuda_main.o cuda/cuda_conv_layer.o cuda/cuda_dense_layer.o cuda/cuda_pool_layer.o cuda/cuda_activ_functions.o"
		USE_CUDA=1
		export USE_CUDA=1
		echo USE_CUDA
	fi

	if [ $i = "OPEN_MP" ]
	then
		arg="$arg -D OPEN_MP -fopenmp"
		export USE_OPENMP=1
		echo USE_OPENMP
	fi

	if [ $i = "BLAS" ]
	then
		arg="$arg -D BLAS -lopenblas -I $openblas_include_dir -L $openblas_lib_dir"
		blas_src="blas_dense_layer.c blas_conv_layer.c" 
		blas_obj="blas/blas_dense_layer.o blas/blas_conv_layer.o"
		USE_BLAS=1
		export USE_BLAS=1
		echo USE_BLAS
	fi

	if [ $i = "LPTHREAD" ]
	then
		arg="$arg -lpthread"
		echo USE_LPTHREAD
	fi
	
	if [ $i = "PY_INTERF" ]
	then
		PY_INTERF=1
		echo BUILD_PY_INTERF
	fi
done

cd ./src

if [ $USE_CUDA ]
then

#compiling the cuda part if needed
cd ./cuda
nvcc --compiler-bindir $gcc_compile_dir -Xcompiler "$compile_opt" \
-O3 -c $cuda_src $cuda_arg $defines_variables -lm
echo "#####  End of CUDA compilation  #####"
cd ..
fi

if [ $USE_BLAS ]
then

cd ./blas
gcc $compile_opt -std=c99 -c \
../defs.h ../prototypes.h ../structs.h $blas_src -lm $arg $defines_variables
echo "#####  End of BLAS compilation  #####"
cd ..
fi

cd ./naiv
gcc $compile_opt -std=c99 -c \
../defs.h ../prototypes.h ../structs.h naiv_dense_layer.c naiv_conv_layer.c naiv_pool_layer.c -lm $arg $defines_variables
cd ..

#compiling all the program
gcc $compile_opt -std=c99 -c \
defs.h prototypes.h structs.h main.c conv_layer.c dense_layer.c pool_layer.c activ_functions.c initializers.c vars.c auxil.c -lm $arg $defines_variables
echo "#####  End of main program compilation  #####"

#linking the main program (with cuda if needed)
gcc $compile_opt -std=c99 -o \
../main main.o $cuda_obj $blas_obj conv_layer.o dense_layer.o pool_layer.o activ_functions.o initializers.o vars.o auxil.o naiv/naiv_dense_layer.o naiv/naiv_conv_layer.o naiv/naiv_pool_layer.o -lm $arg $defines_variables
echo "#####  End of link edition and executable creation  #####"

#rm *.o *.gch

if [ $PY_INTERF ]
then
rm -rf ./build/

python3 python_module_setup.py build
echo "#####  End of Python3 interface build  #####"
fi
cd ..








