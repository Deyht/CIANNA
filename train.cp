######################################################
#               EDIT FOR YOUR INSTALL                #
######################################################

gcc_compile_dir="/usr/bin/gcc"
openblas_include_dir="/opt/OpenBLAS/include/"
openblas_lib_dir="/opt/OpenBLAS/lib"
cuda_lib_path="/usr/local/cuda-10.1/lib64"
compile_opt="-O3 -Wall -Werror -Wno-unused-result -fmax-errors=2 -fbounds-check"

######################################################


for i in $*
do
	if [ $i  = "CUDA" ]
	then
		cuda_arg="$cuda_arg -D CUDA -D comp_CUDA -lcublas -lcudart "
		arg="$arg -D CUDA -lcublas -lcudart -L $cuda_lib_path "
		cuda_obj="cuda_main.o cuda_conv_layer.o cuda_dense_layer.o cuda_pool_layer.o cuda_activ_functions.o"
		USE_CUDA=1
		echo USE_CUDA
	fi

	if [ $i = "OPEN_MP" ]
	then
		arg="$arg -D OPEN_MP -fopenmp"
		echo USE_OPENMP
	fi

	if [ $i = "BLAS" ]
	then
		arg="$arg -D BLAS -lopenblas -I $openblas_include_dir -L $openblas_lib_dir "
		echo USE_BLAS
	fi

	if [ $i = "LPTHREAD" ]
	then
		arg="$arg -lpthread"
		echo USE_LPTHREAD
	fi
done

cd ./src

if [ $USE_CUDA ]
then
#compiling the cuda part if needed
nvcc --compiler-bindir $gcc_compile_dir -Xcompiler "$compile_opt" \
-O3 -c cuda_main.cu cuda_activ_functions.cu cuda_conv_layer.cu cuda_pool_layer.cu cuda_dense_layer.cu $cuda_arg -lm
fi
echo "#####  End of CUDA compilation  #####"


#compiling all the program
gcc $compile_opt -std=c99 -c \
defs.h prototypes.h structs.h main.c conv_layer.c dense_layer.c pool_layer.c activ_functions.c initializers.c -lm $arg
echo "#####  End of main program compilation  #####"

#linking the main program (with cuda if needed)
gcc $compile_opt -std=c99 -o \
../main $cuda_obj main.o conv_layer.o dense_layer.o pool_layer.o activ_functions.o initializers.o -lm $arg
echo "#####  End of link edition and executable creation  #####"

rm *.o *.gch

cd ..








